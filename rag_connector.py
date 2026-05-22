"""
================================================================================
  RAG CONNECTOR — ChromaDB · nomic-embed (Ollama) · BM25 Hybrid Retrieval
================================================================================

AUTHOR  : Senior AI Architect (Multimodal RAG)
VERSION : 1.0.0

PURPOSE
-------
This module is the bridge between the MultimodalRAGPipeline (ingestion) and
your live RAG system (retrieval + generation).  It provides:

  1. OllamaEmbedder        — wraps the local nomic-embed-text model via Ollama
                             to produce dense vector embeddings.

  2. BM25Index             — a lightweight in-process BM25 sparse index built
                             on top of `rank_bm25`.  Persisted to disk as JSON
                             so it survives restarts.

  3. ChromaStore           — manages a ChromaDB collection: upsert, query,
                             delete.  Stores all chunk fields as metadata.

  4. HybridRetriever       — fuses dense (ChromaDB cosine) + sparse (BM25)
                             scores via Reciprocal Rank Fusion (RRF), then
                             returns a unified ranked result list.

  5. RAGConnector          — top-level façade.  Call .index() to ingest a file
                             end-to-end, and .query() to retrieve + generate.

================================================================================
  PREREQUISITES
================================================================================

pip install chromadb rank-bm25 ollama

Ollama setup (already done if you ran the pipeline):
  ollama pull nomic-embed-text   ← embedding model (~274 MB)
  ollama pull llama3             ← generation model (already pulled)
  ollama serve                   ← must be running

================================================================================
  QUICK START
================================================================================

  from multimodal_rag_pipeline import MultimodalRAGPipeline
  from rag_connector import RAGConnector

  rag = RAGConnector()                    # default settings
  rag.index("quarterly_report.pdf")       # ingest any file
  rag.index("meeting_recording.mp3")      # works for all modalities

  results = rag.query("What were the main revenue drivers in Q3?", top_k=5)
  for r in results:
      print(r["answer"])

================================================================================
  ARCHITECTURE — data flow
================================================================================

  .index(file_path)
      │
      ├─ MultimodalRAGPipeline.ingest()
      │       (DataRouter → Processor → Unifier → ChunkingAgent
      │        → ContextualRetriever)
      │       returns List[ChunkDict]
      │
      ├─ OllamaEmbedder.embed(chunk["contextualized_text"])
      │       returns List[float]  (768-dim nomic-embed vector)
      │
      ├─ ChromaStore.upsert(chunk, embedding)
      │       stores vector + all metadata in ChromaDB
      │
      └─ BM25Index.add(chunk)
              adds tokenised chunk_text to sparse index (persisted to disk)

  .query(question, top_k)
      │
      ├─ OllamaEmbedder.embed(question)          → query vector
      │
      ├─ ChromaStore.query(vector, top_k * 2)    → dense candidates
      │
      ├─ BM25Index.query(question, top_k * 2)    → sparse candidates
      │
      ├─ HybridRetriever.fuse(dense, sparse)     → RRF-ranked list
      │       (Reciprocal Rank Fusion: score = Σ 1/(k + rank_i))
      │
      └─ _generate(question, top_chunks)         → answer via llama3

================================================================================
"""

from __future__ import annotations

import json
import logging
import math
import re
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("RAGConnector")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


# ══════════════════════════════════════════════════════════════════════════════
#  1. OLLAMA EMBEDDER
# ══════════════════════════════════════════════════════════════════════════════

class OllamaEmbedder:
    """
    Generates dense vector embeddings using **nomic-embed-text** (or any other
    model available in your local Ollama instance).

    nomic-embed-text produces 768-dimensional embeddings and is specifically
    optimised for retrieval tasks — it outperforms OpenAI ada-002 on MTEB
    while running fully locally.

    Pull model : ollama pull nomic-embed-text
    Alt model  : ollama pull mxbai-embed-large  (1024-dim, higher quality)
    """

    DEFAULT_MODEL = "nomic-embed-text"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self._model = model
        self._validate_ollama()

    def _validate_ollama(self) -> None:
        """Fail fast if ollama library or server is unavailable."""
        try:
            import ollama  # type: ignore
            # Quick ping — list local models
            ollama.list()
        except ImportError as exc:
            raise RuntimeError(
                "ollama not installed. Run: pip install ollama"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach Ollama server: {exc}\n"
                "Make sure 'ollama serve' is running on http://localhost:11434"
            ) from exc

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Parameters
        ----------
        text : str
            The text to embed.  For indexing, pass chunk["contextualized_text"].
            For querying, pass the raw user question.

        Returns
        -------
        list[float]
            Dense embedding vector (768-dim for nomic-embed-text).
        """
        import ollama  # type: ignore

        # Truncate silently if text is extremely long (model max is ~8192 tokens)
        if len(text) > 32_000:
            logger.warning(
                "[OllamaEmbedder] Text truncated from %d to 32 000 chars.", len(text)
            )
            text = text[:32_000]

        try:
            response = ollama.embeddings(model=self._model, prompt=text)
            return response["embedding"]
        except Exception as exc:
            raise RuntimeError(
                f"Embedding call failed: {exc}. "
                f"Is 'ollama pull {self._model}' done?"
            ) from exc

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts.  Calls are sequential because Ollama does not
        expose a native batch endpoint; for GPU inference the bottleneck is
        usually the model, not the HTTP round-trip.
        """
        return [self.embed(t) for t in texts]


# ══════════════════════════════════════════════════════════════════════════════
#  2. BM25 SPARSE INDEX  (persisted to disk)
# ══════════════════════════════════════════════════════════════════════════════

class BM25Index:
    """
    Lightweight BM25 sparse retrieval index backed by **rank_bm25**.

    BM25 excels at exact keyword matching — the complement of dense vector
    search which handles semantic similarity.  Together they form a hybrid
    retrieval system with significantly higher recall than either alone.

    The index is persisted as a JSON file so it survives process restarts.
    On load, the BM25 object is reconstructed from the stored corpus.

    Install : pip install rank-bm25
    """

    def __init__(self, persist_path: str | Path = "bm25_index.json") -> None:
        self._path  : Path              = Path(persist_path)
        # Each entry: {"id": str, "text": str, "tokens": list[str]}
        self._corpus: list[dict]        = []
        self._bm25  : Any               = None  # rank_bm25.BM25Okapi instance
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load corpus from disk and rebuild the BM25 object."""
        if self._path.exists():
            try:
                self._corpus = json.loads(self._path.read_text(encoding="utf-8"))
                self._rebuild()
                logger.info(
                    "[BM25Index] Loaded %d documents from '%s'.",
                    len(self._corpus),
                    self._path,
                )
            except Exception as exc:
                logger.warning("[BM25Index] Could not load index: %s — starting fresh.", exc)
                self._corpus = []

    def _save(self) -> None:
        """Persist the corpus to disk."""
        self._path.write_text(
            json.dumps(self._corpus, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _rebuild(self) -> None:
        """Reconstruct the BM25Okapi object from the current corpus."""
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as exc:
            raise RuntimeError("rank-bm25 not installed. Run: pip install rank-bm25") from exc

        if self._corpus:
            tokenised = [doc["tokens"] for doc in self._corpus]
            self._bm25 = BM25Okapi(tokenised)
        else:
            self._bm25 = None

    # ── Public API ────────────────────────────────────────────────────────────

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        Simple whitespace + punctuation tokenizer.
        Lowercases, removes punctuation, drops tokens < 2 chars.
        """
        tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
        return tokens

    def add(self, doc_id: str, text: str) -> None:
        """
        Add a document to the index.

        Parameters
        ----------
        doc_id : str
            Unique identifier (same as the ChromaDB document ID).
        text   : str
            The text to index.  Use chunk["contextualized_text"] for best recall.
        """
        tokens = self.tokenize(text)
        self._corpus.append({"id": doc_id, "text": text, "tokens": tokens})
        self._rebuild()
        self._save()

    def add_batch(self, docs: list[tuple[str, str]]) -> None:
        """
        Add multiple (doc_id, text) pairs efficiently in one rebuild + save.
        """
        for doc_id, text in docs:
            tokens = self.tokenize(text)
            self._corpus.append({"id": doc_id, "text": text, "tokens": tokens})
        self._rebuild()
        self._save()
        logger.info("[BM25Index] Added %d documents. Total: %d.", len(docs), len(self._corpus))

    def query(self, question: str, top_k: int = 10) -> list[dict]:
        """
        Retrieve the top-k documents by BM25 score.

        Returns
        -------
        list[dict]
            Each dict: {"id": str, "score": float, "rank": int}
            Ordered by descending BM25 score.
        """
        if self._bm25 is None or not self._corpus:
            return []

        query_tokens = self.tokenize(question)
        scores       = self._bm25.get_scores(query_tokens)

        # Pair each document with its score and sort
        ranked = sorted(
            zip(scores, self._corpus),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        return [
            {"id": doc["id"], "score": float(score), "rank": i + 1}
            for i, (score, doc) in enumerate(ranked)
        ]

    def delete(self, doc_id: str) -> None:
        """Remove a document from the index by ID."""
        before = len(self._corpus)
        self._corpus = [d for d in self._corpus if d["id"] != doc_id]
        if len(self._corpus) < before:
            self._rebuild()
            self._save()

    def delete_by_source(self, source: str) -> int:
        """Remove all documents whose text contains the source filename."""
        before = len(self._corpus)
        self._corpus = [
            d for d in self._corpus
            if source not in d.get("text", "")
        ]
        removed = before - len(self._corpus)
        if removed:
            self._rebuild()
            self._save()
        return removed

    @property
    def count(self) -> int:
        return len(self._corpus)


# ══════════════════════════════════════════════════════════════════════════════
#  3. CHROMA STORE
# ══════════════════════════════════════════════════════════════════════════════

class ChromaStore:
    """
    Manages a **ChromaDB** collection for dense vector storage and retrieval.

    ChromaDB is an embedded vector database — no server process required.
    Data is persisted to a local directory (`./chroma_db` by default).

    Each chunk is stored with:
      • The embedding of `contextualized_text`  (for semantic search)
      • `chunk_text` as the document content     (what gets shown to users)
      • All other chunk fields as metadata       (filterable)

    Install : pip install chromadb
    Docs    : https://docs.trychroma.com
    """

    DEFAULT_COLLECTION = "multimodal_rag"
    DEFAULT_PERSIST_DIR = "./chroma_db"

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        persist_dir    : str = DEFAULT_PERSIST_DIR,
    ) -> None:
        try:
            import chromadb  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "chromadb not installed. Run: pip install chromadb"
            ) from exc

        self._client = chromadb.PersistentClient(path=persist_dir)

        # get_or_create: safe to call on every startup
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            # cosine distance is standard for normalised embeddings
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "[ChromaStore] Collection '%s' ready. %d documents stored.",
            collection_name,
            self._collection.count(),
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        doc_id   : str,
        embedding: list[float],
        chunk    : dict,
    ) -> None:
        """
        Insert or update a single chunk.

        Parameters
        ----------
        doc_id    : str         Unique ID (used for deduplication).
        embedding : list[float] Dense vector of contextualized_text.
        chunk     : dict        Full ChunkDict from the pipeline.
        """
        # ChromaDB metadata values must be str | int | float | bool.
        # Convert lists (keywords) to comma-separated strings.
        metadata = {
            "source"   : str(chunk.get("source", "")),
            "modality" : str(chunk.get("modality", "")),
            "summary"  : str(chunk.get("summary", "")),
            "keywords" : ", ".join(chunk.get("keywords", [])),
            "context"  : str(chunk.get("context", "")),
        }

        self._collection.upsert(
            ids        =[doc_id],
            embeddings =[embedding],
            documents  =[chunk.get("chunk_text", "")],  # shown in results
            metadatas  =[metadata],
        )

    def upsert_batch(
        self,
        doc_ids   : list[str],
        embeddings: list[list[float]],
        chunks    : list[dict],
    ) -> None:
        """Batch upsert — much faster than looping upsert() for many chunks."""
        metadatas = [
            {
                "source"   : str(c.get("source", "")),
                "modality" : str(c.get("modality", "")),
                "summary"  : str(c.get("summary", "")),
                "keywords" : ", ".join(c.get("keywords", [])),
                "context"  : str(c.get("context", "")),
            }
            for c in chunks
        ]
        documents = [c.get("chunk_text", "") for c in chunks]

        self._collection.upsert(
            ids        =doc_ids,
            embeddings =embeddings,
            documents  =documents,
            metadatas  =metadatas,
        )
        logger.info("[ChromaStore] Upserted %d chunks.", len(doc_ids))

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        embedding : list[float],
        top_k     : int = 10,
        where     : dict | None = None,
    ) -> list[dict]:
        """
        Retrieve top-k chunks by cosine similarity.

        Parameters
        ----------
        embedding : list[float]   Query vector.
        top_k     : int           Number of results to return.
        where     : dict | None   Optional ChromaDB metadata filter.
                                  e.g. {"modality": "Document"}

        Returns
        -------
        list[dict]
            Each dict: {"id", "document", "metadata", "distance", "rank"}
        """
        kwargs: dict = {
            "query_embeddings": [embedding],
            "n_results"       : min(top_k, self._collection.count() or 1),
            "include"         : ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        output = []
        for i, (doc_id, doc, meta, dist) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            output.append({
                "id"       : doc_id,
                "document" : doc,       # chunk_text
                "metadata" : meta,
                "distance" : dist,
                "score"    : 1 - dist,  # cosine similarity (0–1, higher = better)
                "rank"     : i + 1,
            })

        return output

    def get_by_id(self, doc_id: str) -> dict | None:
        """Fetch a single chunk by its ID."""
        result = self._collection.get(ids=[doc_id], include=["documents", "metadatas"])
        if result["ids"]:
            return {
                "id"      : result["ids"][0],
                "document": result["documents"][0],
                "metadata": result["metadatas"][0],
            }
        return None

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_by_source(self, source_filename: str) -> int:
        """
        Delete all chunks originating from a specific source file.
        Useful for re-ingesting an updated document.
        """
        results = self._collection.get(
            where={"source": source_filename},
            include=[],
        )
        ids = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
            logger.info(
                "[ChromaStore] Deleted %d chunks for source '%s'.",
                len(ids),
                source_filename,
            )
        return len(ids)

    @property
    def count(self) -> int:
        return self._collection.count()


# ══════════════════════════════════════════════════════════════════════════════
#  4. HYBRID RETRIEVER  (Reciprocal Rank Fusion)
# ══════════════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    Fuses dense (ChromaDB) and sparse (BM25) rankings using
    **Reciprocal Rank Fusion (RRF)**.

    WHY RRF?
    --------
    Normalising raw scores across different retrieval systems is unreliable
    (BM25 scores are unbounded; cosine similarities are 0–1).  RRF sidesteps
    this by using only *rank positions*, making it robust to score scale
    differences and consistently outperforming score-based fusion in benchmarks.

    FORMULA
    -------
        RRF(d) = Σ_i  1 / (k + rank_i(d))

    where k=60 is the standard smoothing constant (Cormack et al., 2009).
    Documents that don't appear in a list are treated as rank = ∞ (score = 0).

    The final list is sorted by descending RRF score.
    """

    RRF_K = 60  # standard smoothing constant

    def fuse(
        self,
        dense_results : list[dict],
        sparse_results: list[dict],
        top_k         : int = 5,
    ) -> list[dict]:
        """
        Combine dense and sparse result lists into a single ranked list.

        Parameters
        ----------
        dense_results  : list[dict]   Output of ChromaStore.query()
        sparse_results : list[dict]   Output of BM25Index.query()
        top_k          : int          How many fused results to return.

        Returns
        -------
        list[dict]
            Merged, deduplicated, RRF-ranked results.  Each dict has:
            {"id", "document", "metadata", "rrf_score",
             "dense_rank", "sparse_rank", "rank"}
        """
        # Build lookup: id → metadata/document from ChromaDB
        # (BM25 only has IDs; we need ChromaDB data for display)
        chroma_lookup: dict[str, dict] = {r["id"]: r for r in dense_results}

        # Compute RRF scores
        scores: dict[str, float] = {}

        for result in dense_results:
            doc_id = result["id"]
            rank   = result["rank"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.RRF_K + rank)

        for result in sparse_results:
            doc_id = result["id"]
            rank   = result["rank"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.RRF_K + rank)

        # Sort by RRF score descending
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

        # Build output — enrich with metadata from ChromaDB where available
        output = []
        for final_rank, doc_id in enumerate(sorted_ids, 1):
            chroma  = chroma_lookup.get(doc_id, {})
            sparse  = next((r for r in sparse_results if r["id"] == doc_id), {})
            dense   = next((r for r in dense_results  if r["id"] == doc_id), {})

            output.append({
                "id"          : doc_id,
                "document"    : chroma.get("document", ""),   # chunk_text
                "metadata"    : chroma.get("metadata", {}),
                "rrf_score"   : round(scores[doc_id], 6),
                "dense_rank"  : dense.get("rank"),
                "sparse_rank" : sparse.get("rank"),
                "rank"        : final_rank,
            })

        return output


# ══════════════════════════════════════════════════════════════════════════════
#  5. RAG CONNECTOR  (top-level façade)
# ══════════════════════════════════════════════════════════════════════════════

class RAGConnector:
    """
    Top-level façade that wires together:
      MultimodalRAGPipeline → OllamaEmbedder → ChromaStore + BM25Index
                                             → HybridRetriever → llama3

    TWO PUBLIC METHODS
    ------------------
    .index(file_path)          — ingest a file end-to-end
    .query(question, top_k)    — hybrid retrieve + LLM answer

    USAGE
    -----
    >>> from rag_connector import RAGConnector
    >>> rag = RAGConnector()
    >>> rag.index("report.pdf")
    >>> rag.index("meeting.mp3")
    >>> results = rag.query("What were the key risks discussed?", top_k=5)
    >>> print(results["answer"])
    >>> for chunk in results["sources"]:
    ...     print(chunk["metadata"]["source"], "—", chunk["metadata"]["summary"])
    """

    GENERATION_MODEL = "llama3"

    # System prompt for the final answer generation step
    GENERATION_SYSTEM_PROMPT = """\
You are a precise, factual assistant answering questions based ONLY on the \
provided source chunks. Follow these rules:
1. Base your answer SOLELY on the provided chunks. Do not use prior knowledge.
2. If the chunks do not contain enough information, say so explicitly.
3. Cite the source of each claim using the format [Source: filename].
4. Be concise but complete. Use bullet points for multi-part answers.
5. Never fabricate facts, dates, names, or numbers."""

    def __init__(
        self,
        # Pipeline settings
        contextual_retrieval: bool = True,
        context_max_workers : int  = 4,
        # Embedding
        embed_model         : str  = OllamaEmbedder.DEFAULT_MODEL,
        # ChromaDB
        chroma_collection   : str  = ChromaStore.DEFAULT_COLLECTION,
        chroma_persist_dir  : str  = ChromaStore.DEFAULT_PERSIST_DIR,
        # BM25
        bm25_persist_path   : str  = "bm25_index.json",
        # Generation
        generation_model    : str  = GENERATION_MODEL,
    ) -> None:
        # Lazy import the pipeline so this file can be used standalone
        # even if the pipeline file is not yet in the Python path.
        try:
            from multimodal_rag_pipeline import MultimodalRAGPipeline  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "multimodal_rag_pipeline.py must be in the same directory "
                "or on PYTHONPATH."
            ) from exc

        self._pipeline = MultimodalRAGPipeline(
            contextual_retrieval=contextual_retrieval,
            context_max_workers=context_max_workers,
        )
        self._embedder   = OllamaEmbedder(model=embed_model)
        self._chroma     = ChromaStore(
            collection_name=chroma_collection,
            persist_dir=chroma_persist_dir,
        )
        self._bm25       = BM25Index(persist_path=bm25_persist_path)
        self._retriever  = HybridRetriever()
        self._gen_model  = generation_model

        logger.info(
            "[RAGConnector] Ready. ChromaDB: %d docs | BM25: %d docs",
            self._chroma.count,
            self._bm25.count,
        )

    # ── INDEXING ──────────────────────────────────────────────────────────────

    def index(
        self,
        file_path          : str | Path,
        delete_existing    : bool = True,
    ) -> int:
        """
        Ingest a file end-to-end and store its chunks.

        Steps
        -----
        1. Run MultimodalRAGPipeline (route → process → unify → chunk → contextualise)
        2. Embed each chunk's `contextualized_text` via nomic-embed-text
        3. Upsert into ChromaDB (dense) and BM25Index (sparse)

        Parameters
        ----------
        file_path       : Path to any supported file.
        delete_existing : If True, delete all previously stored chunks from
                          this source before re-indexing.  Prevents duplicates
                          when re-ingesting an updated file.

        Returns
        -------
        int
            Number of chunks indexed.
        """
        path = Path(file_path)
        logger.info("═══ [RAGConnector] Indexing '%s' ═══", path.name)

        # ── Step 1: Delete old chunks if re-indexing ──────────────────────
        if delete_existing:
            n_deleted = self._chroma.delete_by_source(path.name)
            self._bm25.delete_by_source(path.name)
            if n_deleted:
                logger.info("[RAGConnector] Removed %d stale chunks.", n_deleted)

        # ── Step 2: Run the ingestion pipeline ────────────────────────────
        chunks = self._pipeline.ingest(path)
        if not chunks:
            logger.warning("[RAGConnector] Pipeline returned 0 chunks for '%s'.", path.name)
            return 0

        # Attach source filename to each chunk for metadata storage
        for chunk in chunks:
            chunk.setdefault("source", path.name)

        # ── Step 3: Embed all contextualized_text strings ─────────────────
        logger.info("[RAGConnector] Embedding %d chunks…", len(chunks))

        texts_to_embed = [
            # Prefer contextualized_text (context + chunk); fall back to chunk_text
            chunk.get("contextualized_text") or chunk.get("chunk_text", "")
            for chunk in chunks
        ]
        embeddings = self._embedder.embed_batch(texts_to_embed)

        # ── Step 4: Generate stable unique IDs ────────────────────────────
        # Format: source_filename::chunk_index  (deterministic for dedup)
        doc_ids = [
            f"{path.stem}::{i:04d}"
            for i in range(len(chunks))
        ]

        # ── Step 5: Batch upsert into ChromaDB ────────────────────────────
        self._chroma.upsert_batch(
            doc_ids   =doc_ids,
            embeddings=embeddings,
            chunks    =chunks,
        )

        # ── Step 6: Batch add to BM25 ─────────────────────────────────────
        bm25_docs = [
            (doc_id, chunk.get("contextualized_text") or chunk.get("chunk_text", ""))
            for doc_id, chunk in zip(doc_ids, chunks)
        ]
        self._bm25.add_batch(bm25_docs)

        logger.info(
            "═══ [RAGConnector] Indexed '%s' → %d chunks stored ═══",
            path.name,
            len(chunks),
        )
        return len(chunks)

    # ── QUERYING ──────────────────────────────────────────────────────────────

    def query(
        self,
        question        : str,
        top_k           : int = 5,
        modality_filter : str | None = None,
    ) -> dict:
        """
        Retrieve relevant chunks and generate an answer.

        Parameters
        ----------
        question        : str           The user's natural language question.
        top_k           : int           Final number of chunks to pass to the LLM.
        modality_filter : str | None    Optional filter: "Document" | "Audio" |
                                        "Image" | "Video" | "Code"

        Returns
        -------
        dict with keys:
          "answer"  : str          LLM-generated answer grounded in the chunks.
          "sources" : list[dict]   The top-k chunks used (with metadata).
          "question": str          Echo of the original question.
        """
        logger.info("[RAGConnector] Query: '%s'", question[:80])

        if self._chroma.count == 0:
            return {
                "answer"  : "The knowledge base is empty. Please index some files first.",
                "sources" : [],
                "question": question,
            }

        # ── Step 1: Embed the question ────────────────────────────────────
        query_embedding = self._embedder.embed(question)

        # ── Step 2: Dense retrieval (ChromaDB) ───────────────────────────
        where = {"modality": modality_filter} if modality_filter else None
        dense_results = self._chroma.query(
            embedding=query_embedding,
            top_k=top_k * 2,     # over-retrieve for RRF fusion
            where=where,
        )

        # ── Step 3: Sparse retrieval (BM25) ──────────────────────────────
        sparse_results = self._bm25.query(question, top_k=top_k * 2)

        # ── Step 4: Hybrid fusion (RRF) ───────────────────────────────────
        fused = self._retriever.fuse(
            dense_results =dense_results,
            sparse_results=sparse_results,
            top_k=top_k,
        )

        logger.info(
            "[RAGConnector] Retrieval: %d dense | %d sparse | %d fused",
            len(dense_results),
            len(sparse_results),
            len(fused),
        )

        # ── Step 5: Generate answer via llama3 ───────────────────────────
        answer = self._generate(question, fused)

        return {
            "answer"  : answer,
            "sources" : fused,
            "question": question,
        }

    # ── GENERATION ────────────────────────────────────────────────────────────

    def _generate(self, question: str, chunks: list[dict]) -> str:
        """
        Build a grounded prompt from retrieved chunks and call llama3 to
        generate a final answer.

        The prompt structure follows the RAG pattern:
          SYSTEM: instructions + grounding rules
          USER:   retrieved context + question
        """
        import ollama  # type: ignore

        # Build context block from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source   = chunk.get("metadata", {}).get("source", "unknown")
            modality = chunk.get("metadata", {}).get("modality", "")
            doc_text = chunk.get("document", "")  # chunk_text from ChromaDB
            context_parts.append(
                f"[{i}] Source: {source} ({modality})\n{doc_text}"
            )

        context_block = "\n\n---\n\n".join(context_parts)

        user_message = (
            f"Here are the relevant source chunks:\n\n"
            f"{context_block}\n\n"
            f"---\n\n"
            f"Question: {question}\n\n"
            f"Answer (cite sources using [Source: filename]):"
        )

        try:
            response = ollama.chat(
                model=self._gen_model,
                messages=[
                    {"role": "system", "content": self.GENERATION_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                options={
                    "temperature": 0.2,   # low temperature for factual answers
                    "num_ctx"    : 8192,
                },
            )
            return response["message"]["content"].strip()
        except Exception as exc:
            logger.error("[RAGConnector] Generation failed: %s", exc)
            return (
                f"⚠️ Answer generation failed: {exc}\n\n"
                f"Retrieved {len(chunks)} relevant chunks — "
                "check logs for details."
            )

    # ── UTILITIES ─────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Return a snapshot of the current index sizes."""
        return {
            "chroma_docs": self._chroma.count,
            "bm25_docs"  : self._bm25.count,
        }

    def delete_source(self, source_filename: str) -> dict:
        """Remove all chunks from a specific source file."""
        n_chroma = self._chroma.delete_by_source(source_filename)
        n_bm25   = self._bm25.delete_by_source(source_filename)
        logger.info(
            "[RAGConnector] Deleted '%s': %d Chroma | %d BM25 chunks.",
            source_filename, n_chroma, n_bm25,
        )
        return {"chroma_deleted": n_chroma, "bm25_deleted": n_bm25}


# ══════════════════════════════════════════════════════════════════════════════
#  CLI / DEMO
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Interactive demo CLI.

    Usage
    -----
      # Index a file
      python rag_connector.py index report.pdf

      # Query the knowledge base
      python rag_connector.py query "What were the main revenue drivers in Q3?"

      # Query with modality filter
      python rag_connector.py query "Summarise the meeting" --modality Audio

      # Show index stats
      python rag_connector.py stats

      # Delete a source
      python rag_connector.py delete report.pdf
    """
    import sys

    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd  = args[0].lower()
    rag  = RAGConnector()

    if cmd == "index":
        if len(args) < 2:
            print("Usage: python rag_connector.py index <file_path>")
            sys.exit(1)
        n = rag.index(args[1])
        print(f"✓ Indexed {n} chunks from '{args[1]}'.")

    elif cmd == "query":
        if len(args) < 2:
            print("Usage: python rag_connector.py query \"<question>\" [--modality <type>]")
            sys.exit(1)

        question = args[1]
        modality = None
        if "--modality" in args:
            idx      = args.index("--modality")
            modality = args[idx + 1] if idx + 1 < len(args) else None

        result = rag.query(question, top_k=5, modality_filter=modality)

        print(f"\n{'═' * 60}")
        print(f"Q: {result['question']}")
        print(f"{'═' * 60}")
        print(f"\n{result['answer']}\n")
        print(f"{'─' * 60}")
        print("Sources used:")
        for src in result["sources"]:
            meta = src.get("metadata", {})
            print(
                f"  [{src['rank']}] {meta.get('source', '?')} "
                f"({meta.get('modality', '?')}) "
                f"— RRF: {src['rrf_score']:.4f} "
                f"[dense: #{src['dense_rank']} | sparse: #{src['sparse_rank']}]"
            )

    elif cmd == "stats":
        s = rag.stats
        print(f"ChromaDB documents : {s['chroma_docs']}")
        print(f"BM25 documents     : {s['bm25_docs']}")

    elif cmd == "delete":
        if len(args) < 2:
            print("Usage: python rag_connector.py delete <source_filename>")
            sys.exit(1)
        result = rag.delete_source(args[1])
        print(f"Deleted: {result}")

    else:
        print(f"Unknown command '{cmd}'. Use: index | query | stats | delete")
        sys.exit(1)


if __name__ == "__main__":
    main()

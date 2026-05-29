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
from datetime import datetime, timezone
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
        # indexed_at is ISO-8601 for display; indexed_ts is a Unix int for
        # range filtering (ChromaDB's $gte / $lte work numerically on ints).
        metadata = {
            "source"     : str(chunk.get("source", "")),
            "modality"   : str(chunk.get("modality", "")),
            "summary"    : str(chunk.get("summary", "")),
            "keywords"   : ", ".join(chunk.get("keywords", [])),
            "context"    : str(chunk.get("context", "")),
            "indexed_at" : str(chunk.get("indexed_at", "")),
            "indexed_ts" : int(chunk.get("indexed_ts", 0)),
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
                "source"     : str(c.get("source", "")),
                "modality"   : str(c.get("modality", "")),
                "summary"    : str(c.get("summary", "")),
                "keywords"   : ", ".join(c.get("keywords", [])),
                "context"    : str(c.get("context", "")),
                "indexed_at" : str(c.get("indexed_at", "")),
                "indexed_ts" : int(c.get("indexed_ts", 0)),
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

    def filter_ids(self, ids: list[str], where: dict | None) -> list[str]:
        """
        Return the subset of *ids* whose stored metadata matches *where*.

        Used to apply metadata filters to BM25 (sparse) results, since BM25
        itself is metadata-blind. Dense retrieval already filters natively
        via ChromaDB's `where` parameter.
        """
        if not ids or not where:
            return ids
        result = self._collection.get(ids=ids, where=where, include=[])
        return result.get("ids", [])

    def list_distinct(self, field: str) -> list[str]:
        """
        Return the sorted list of distinct values stored for a metadata field.

        Used to populate the UI's filter dropdowns (sources, modalities).
        ChromaDB has no native DISTINCT, so we fetch all metadata and
        deduplicate in Python — fine for indexes up to ~10⁵ chunks.
        """
        result = self._collection.get(include=["metadatas"])
        values: set[str] = set()
        for meta in result.get("metadatas", []) or []:
            v = meta.get(field)
            if v:
                values.add(str(v))
        return sorted(values)

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
#  4. CROSS-ENCODER RERANKER  (Second-stage precision)
# ══════════════════════════════════════════════════════════════════════════════

class CrossEncoderReranker:
    """
    Reranks retrieved candidates using a cross-encoder model that jointly
    encodes (query, candidate) pairs to judge relevance.

    WHY RERANKING?
    ──────────────
    RRF fusion ranks by position alone — it doesn't actually measure relevance.
    A cross-encoder scores query–chunk pairs end-to-end, catching false positives
    that RRF sometimes promotes.

    EXAMPLE IMPROVEMENT
    ───────────────────
    Query: "What caused the revenue decline?"
    RRF rank 1: "Revenue declined by 12%"         ← mentions revenue, high RRF score
    RRF rank 2: "Revenue drivers: market size"    ← answers the Q, lower RRF score

    Cross-encoder reranks to:
    1. "Revenue drivers: market size"              ← actual cause
    2. "Revenue declined by 12%"                   ← just the fact

    The model **bge-reranker-v2-m3** (from BAAI) runs locally via sentence-transformers
    and is optimized for this task. Adds ~5–10ms per pair, negligible for top-k reranking.

    Install : pip install sentence-transformers
    Model   : bge-reranker-v2-m3 (auto-downloads ~1.1 GB on first run)

    Cost trade-off
    ──────────────
    • Speed: +100–300ms for reranking 20 candidates (acceptable for search)
    • Quality: ~5–15% improvement in top-1 relevance (measured on MTEB benchmarks)
    • Simplicity: Drop-in after RRF; no changes to ingestion
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self._model_name = model
        self._model: Any = None  # lazy-loaded

    def _load_model(self) -> Any:
        """Load and cache the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                ) from exc

            logger.info(
                "[CrossEncoderReranker] Loading %s (first run downloads model)…",
                self._model_name,
            )
            self._model = CrossEncoder(self._model_name, max_length=512)
            logger.info("[CrossEncoderReranker] Model loaded.")

        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank candidates by relevance to the query.

        Parameters
        ----------
        query : str
            The search query.
        candidates : list[dict]
            Retrieved chunks from RRF. Each must have a "document" key.
        top_k : int
            How many reranked results to return.

        Returns
        -------
        list[dict]
            Top-k candidates sorted by cross-encoder score.
            Adds "rerank_score" to each dict.
        """
        if not candidates:
            return candidates

        model = self._load_model()

        # Prepare pairs: (query, candidate_text) for the model
        pairs = [
            [query, cand.get("document", "")]
            for cand in candidates
        ]

        logger.debug("[CrossEncoderReranker] Scoring %d pairs…", len(pairs))

        try:
            # Scores are typically in range [0, 1] (higher = more relevant)
            scores = model.predict(pairs)
        except Exception as exc:
            logger.error("[CrossEncoderReranker] Scoring failed: %s", exc)
            # Degrade gracefully: return original order
            return candidates[:top_k]

        # Attach scores and sort by descending relevance
        scored = [
            {**cand, "rerank_score": float(score)}
            for cand, score in zip(candidates, scores)
        ]
        ranked = sorted(scored, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

        logger.info(
            "[CrossEncoderReranker] Top rerank score: %.3f | Bottom: %.3f",
            ranked[0]["rerank_score"],
            ranked[-1]["rerank_score"],
        )

        # Update rank field to reflect new order
        for i, item in enumerate(ranked, 1):
            item["rank"] = i

        return ranked


# ══════════════════════════════════════════════════════════════════════════════
#  5. HYBRID RETRIEVER  (Reciprocal Rank Fusion)
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
        chroma_store  : 'ChromaStore | None' = None,
    ) -> list[dict]:
        """
        Combine dense and sparse result lists into a single ranked list.

        Parameters
        ----------
        dense_results  : list[dict]   Output of ChromaStore.query()
        sparse_results : list[dict]   Output of BM25Index.query()
        top_k          : int          How many fused results to return.
        chroma_store   : ChromaStore  Reference to ChromaDB store to fetch
                                      sparse-only hits. If None, sparse-only
                                      results will have empty document/metadata.

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

        # For sparse-only results (not in dense lookup), fetch from ChromaDB to avoid
        # passing blank documents to the LLM. This preserves the hybrid retrieval benefit.
        missing_ids = [doc_id for doc_id in sorted_ids if doc_id not in chroma_lookup]
        if missing_ids and chroma_store:
            for doc_id in missing_ids:
                doc = chroma_store.get_by_id(doc_id)
                if doc:
                    chroma_lookup[doc_id] = doc

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
#  6. QUERY REWRITER  (conversational / multi-turn retrieval)
# ══════════════════════════════════════════════════════════════════════════════

class QueryRewriter:
    """
    Rewrites a follow-up question into a self-contained, standalone search query
    using the conversation history.

    WHY THIS MATTERS
    ────────────────
    In a multi-turn chat, follow-up questions are full of references that only
    make sense given the previous turns:

        User: "What were Acme's Q3 revenue drivers?"
        Bot : "Cloud services and enterprise licensing…"
        User: "And how did THAT compare to Q2?"   ← "that" = revenue drivers

    Embedding "And how did that compare to Q2?" retrieves almost nothing useful —
    the pronoun carries no semantic signal. The rewriter resolves the references
    against history first, producing a query the retriever can actually match:

        "How did Acme's Q3 revenue drivers (cloud services and enterprise
         licensing) compare to Q2?"

    This is the standard "condense question" / history-aware retrieval pattern.
    Retrieval uses the rewritten query; the LLM still answers the user's original
    phrasing, with recent history supplied for natural, coherent replies.

    Uses the same local Llama 3 model as generation — no extra dependencies.
    """

    MODEL = "llama3"

    #: How many of the most recent messages to feed the rewriter.
    DEFAULT_HISTORY_TURNS = 6

    SYSTEM_PROMPT = (
        "You rewrite a user's latest message into a single, fully self-contained "
        "search query for a retrieval system.\n\n"
        "RULES:\n"
        "- Resolve all pronouns and references (it, that, they, the previous one) "
        "using the conversation history.\n"
        "- Preserve the original intent. Do NOT answer the question.\n"
        "- If the latest message is already self-contained, return it unchanged.\n"
        "- Output ONLY the rewritten query — no preamble, labels, or quotation marks."
    )

    def __init__(self, model: str = MODEL, history_turns: int = DEFAULT_HISTORY_TURNS) -> None:
        self._model         = model
        self._history_turns = history_turns

    def rewrite(self, question: str, chat_history: list[dict] | None) -> str:
        """
        Produce a standalone query from *question* + *chat_history*.

        Parameters
        ----------
        question     : str
            The user's latest (possibly elliptical) question.
        chat_history : list[dict] | None
            Prior messages, each {"role": "user"|"assistant", "content": str}.
            Should NOT include the current question.

        Returns
        -------
        str
            The rewritten standalone query, or the original question unchanged
            if there is no history or rewriting fails.
        """
        # No history → nothing to resolve; return as-is.
        if not chat_history:
            return question

        import ollama  # type: ignore

        # Build a compact transcript from the most recent turns.
        recent = chat_history[-self._history_turns:]
        transcript = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in recent
        )

        user_message = (
            f"Conversation history:\n{transcript}\n\n"
            f"Latest message: {question}\n\n"
            f"Rewritten standalone query:"
        )

        try:
            response = ollama.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                options={"temperature": 0.0, "num_ctx": 4096},
            )
            rewritten = response["message"]["content"].strip().strip('"')
            # Guard against the model returning an empty string or refusing.
            if rewritten:
                logger.info("[QueryRewriter] '%s' → '%s'", question[:50], rewritten[:50])
                return rewritten
        except Exception as exc:
            logger.warning("[QueryRewriter] Rewrite failed (%s) — using original.", exc)

        return question


# ══════════════════════════════════════════════════════════════════════════════
#  7. RAG CONNECTOR  (top-level façade)
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

    #: How many recent chat messages to include in the generation prompt
    #: for conversational coherence (keeps the context window manageable).
    HISTORY_TURNS_IN_PROMPT = 4

    # System prompt for the final answer generation step
    GENERATION_SYSTEM_PROMPT = """\
You are a precise, factual assistant answering questions based ONLY on the \
provided source chunks. Follow these rules:
1. Base your answer SOLELY on the provided chunks. Do not use prior knowledge.
2. If the chunks do not contain enough information, say so explicitly.
3. Cite EVERY claim with bracketed source numbers matching the chunk list,
   e.g. "Revenue grew 18% [1]." Combine multiple sources as [1, 3] when a
   claim is supported by more than one chunk.
4. When chunks come from DIFFERENT source files, attribute each claim to its
   correct source — do not blend facts across files.
5. Be concise but complete. Use bullet points for multi-part answers.
6. Never fabricate facts, dates, names, or numbers."""

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
        # Reranking (optional)
        use_reranker        : bool = False,
        reranker_model      : str  = CrossEncoderReranker.DEFAULT_MODEL,
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
        self._reranker   = (
            CrossEncoderReranker(model=reranker_model)
            if use_reranker
            else None
        )
        self._rewriter   = QueryRewriter(model=generation_model)
        self._gen_model  = generation_model

        logger.info(
            "[RAGConnector] Ready. ChromaDB: %d docs | BM25: %d docs | Reranker: %s",
            self._chroma.count,
            self._bm25.count,
            "enabled" if self._reranker else "disabled",
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

        return self._store_chunks(chunks, source_name=path.name, id_prefix=path.stem)

    def index_url(self, url: str, delete_existing: bool = True) -> dict:
        """
        Fetch a web page, ingest it, and store its chunks.

        Returns
        -------
        dict: {"source": <page title>, "chunks": <int>}
        """
        logger.info("═══ [RAGConnector] Indexing URL '%s' ═══", url)

        # Run the URL pipeline first so we know the resolved source name (title).
        chunks, source_name = self._pipeline.ingest_url(url)
        if not chunks:
            logger.warning("[RAGConnector] URL pipeline returned 0 chunks for '%s'.", url)
            return {"source": source_name, "chunks": 0}

        # Re-index cleanly: drop any prior chunks from the same source.
        if delete_existing:
            self._chroma.delete_by_source(source_name)
            self._bm25.delete_by_source(source_name)

        n = self._store_chunks(
            chunks, source_name=source_name, id_prefix=self._slugify(source_name)
        )
        return {"source": source_name, "chunks": n}

    @staticmethod
    def _slugify(text: str) -> str:
        """Make a filesystem/ID-safe prefix from arbitrary text."""
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
        return slug[:50] or "web"

    def _store_chunks(
        self,
        chunks    : list[dict],
        source_name: str,
        id_prefix : str,
    ) -> int:
        """
        Shared storage path for both file and URL ingestion:
        stamp timestamps → embed → upsert to ChromaDB → add to BM25.
        """
        # Attach source name and ingestion timestamps to each chunk.
        # indexed_at (ISO) is for display; indexed_ts (Unix int) is what the
        # UI date-range filter uses with ChromaDB's $gte/$lte operators.
        now     = datetime.now(tz=timezone.utc)
        iso_ts  = now.isoformat(timespec="seconds")
        unix_ts = int(now.timestamp())
        for chunk in chunks:
            chunk.setdefault("source", source_name)
            chunk["indexed_at"] = iso_ts
            chunk["indexed_ts"] = unix_ts

        logger.info("[RAGConnector] Embedding %d chunks…", len(chunks))
        texts_to_embed = [
            chunk.get("contextualized_text") or chunk.get("chunk_text", "")
            for chunk in chunks
        ]
        embeddings = self._embedder.embed_batch(texts_to_embed)

        # Stable, deterministic IDs: <prefix>::<index>
        doc_ids = [f"{id_prefix}::{i:04d}" for i in range(len(chunks))]

        self._chroma.upsert_batch(doc_ids=doc_ids, embeddings=embeddings, chunks=chunks)

        bm25_docs = [
            (doc_id, chunk.get("contextualized_text") or chunk.get("chunk_text", ""))
            for doc_id, chunk in zip(doc_ids, chunks)
        ]
        self._bm25.add_batch(bm25_docs)

        logger.info(
            "═══ [RAGConnector] Indexed '%s' → %d chunks stored ═══",
            source_name, len(chunks),
        )
        return len(chunks)

    # ── QUERYING ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_where_filter(
        modality_filter : str | list[str] | None = None,
        source_filter   : str | list[str] | None = None,
        date_from_ts    : int | None = None,
        date_to_ts      : int | None = None,
    ) -> dict | None:
        """
        Build a ChromaDB-compatible `where` clause from independent filter inputs.

        ChromaDB requires a top-level `$and` when combining multiple constraints.
        Single-constraint filters are returned as a plain dict for clarity.
        """
        clauses: list[dict] = []

        def _as_clause(field: str, value: str | list[str]) -> dict:
            if isinstance(value, list):
                value = [v for v in value if v]
                if not value:
                    return {}
                if len(value) == 1:
                    return {field: value[0]}
                return {field: {"$in": value}}
            return {field: value}

        if modality_filter:
            c = _as_clause("modality", modality_filter)
            if c: clauses.append(c)
        if source_filter:
            c = _as_clause("source", source_filter)
            if c: clauses.append(c)
        if date_from_ts is not None:
            clauses.append({"indexed_ts": {"$gte": int(date_from_ts)}})
        if date_to_ts is not None:
            clauses.append({"indexed_ts": {"$lte": int(date_to_ts)}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def query(
        self,
        question        : str,
        top_k           : int = 5,
        modality_filter : str | list[str] | None = None,
        source_filter   : str | list[str] | None = None,
        date_from_ts    : int | None = None,
        date_to_ts      : int | None = None,
        chat_history    : list[dict] | None = None,
        extra_system    : str | None = None,
        memory_notes    : list[str] | None = None,
    ) -> dict:
        """
        Retrieve relevant chunks and generate an answer.

        Parameters
        ----------
        question        : str           The user's natural language question.
        top_k           : int           Final number of chunks to pass to the LLM.
        modality_filter : str | list    Optional. One or more of: "Document",
                                        "Audio", "Image", "Video", "Code".
        source_filter   : str | list    Optional. Restrict to specific source
                                        filenames (e.g. ["report.pdf"]).
        date_from_ts    : int | None    Optional. Only chunks indexed at or
                                        after this Unix timestamp.
        date_to_ts      : int | None    Optional. Only chunks indexed at or
                                        before this Unix timestamp.
        chat_history    : list[dict]    Prior turns for multi-turn retrieval, each
                                        {"role": ..., "content": ...}. Should NOT
                                        include the current question. When provided,
                                        the question is rewritten into a standalone
                                        query before retrieval, and recent history
                                        is given to the LLM for coherent answers.

        Returns
        -------
        dict with keys:
          "answer"           : str          LLM-generated answer grounded in the chunks.
          "sources"          : list[dict]   The top-k chunks used (with metadata).
          "question"         : str          Echo of the original question.
          "search_query"     : str          The (possibly rewritten) query used for retrieval.
        """
        logger.info("[RAGConnector] Query: '%s'", question[:80])

        if self._chroma.count == 0:
            return {
                "answer"      : "The knowledge base is empty. Please index some files first.",
                "sources"     : [],
                "question"    : question,
                "search_query": question,
            }

        # ── Step 0: Rewrite the question for retrieval (multi-turn) ───────
        # Resolves pronouns/references against chat history so follow-ups
        # like "how does that compare to Q2?" retrieve the right chunks.
        search_query = self._rewriter.rewrite(question, chat_history)

        # ── Step 1: Embed the (rewritten) query ───────────────────────────
        query_embedding = self._embedder.embed(search_query)

        # Combined metadata filter (modality / source / date range). Dense
        # retrieval uses this natively; for BM25 we filter the IDs after
        # retrieval since rank_bm25 has no concept of metadata.
        where = self._build_where_filter(
            modality_filter=modality_filter,
            source_filter=source_filter,
            date_from_ts=date_from_ts,
            date_to_ts=date_to_ts,
        )

        # ── Step 2: Dense retrieval (ChromaDB) ───────────────────────────
        dense_results = self._chroma.query(
            embedding=query_embedding,
            top_k=top_k * 2,     # over-retrieve for RRF fusion
            where=where,
        )

        # ── Step 3: Sparse retrieval (BM25) ──────────────────────────────
        # Over-retrieve generously when filtering so the surviving set is
        # still rich enough for RRF fusion + reranking.
        sparse_pool = top_k * 4 if where else top_k * 2
        sparse_results = self._bm25.query(search_query, top_k=sparse_pool)

        # Apply the same metadata filter to BM25 results via ChromaDB lookup.
        if where and sparse_results:
            allowed = set(self._chroma.filter_ids(
                [r["id"] for r in sparse_results], where
            ))
            sparse_results = [r for r in sparse_results if r["id"] in allowed]
            # Renumber ranks after filtering so RRF stays consistent.
            for i, r in enumerate(sparse_results[:top_k * 2], 1):
                r["rank"] = i
            sparse_results = sparse_results[:top_k * 2]

        # ── Step 4: Hybrid fusion (RRF) ───────────────────────────────────
        fused = self._retriever.fuse(
            dense_results =dense_results,
            sparse_results=sparse_results,
            top_k=top_k * 2,  # over-retrieve for reranking
            chroma_store=self._chroma,
        )

        # ── Step 5: Optional reranking (cross-encoder) ─────────────────────
        if self._reranker:
            fused = self._reranker.rerank(search_query, fused, top_k=top_k)

        logger.info(
            "[RAGConnector] Retrieval: %d dense | %d sparse | %d fused" +
            (" | reranked" if self._reranker else ""),
            len(dense_results),
            len(sparse_results),
            len(fused),
        )

        # ── Step 6: Generate answer via llama3 ───────────────────────────
        # Answer the user's ORIGINAL phrasing, with recent history for coherence.
        answer = self._generate(
            question, fused,
            chat_history=chat_history,
            extra_system=extra_system,
            memory_notes=memory_notes,
        )

        return {
            "answer"      : answer,
            "sources"     : fused,
            "question"    : question,
            "search_query": search_query,
        }

    # ── GENERATION ────────────────────────────────────────────────────────────

    def _generate(
        self,
        question: str,
        chunks: list[dict],
        chat_history: list[dict] | None = None,
        extra_system: str | None = None,
        memory_notes: list[str] | None = None,
    ) -> str:
        """
        Build a grounded prompt from retrieved chunks and call llama3 to
        generate a final answer.

        The prompt structure follows the RAG pattern:
          SYSTEM:    base instructions + grounding rules
                     + user's custom system prompt (optional)
                     + persistent memory notes (optional)
          HISTORY:   recent prior turns (optional, for conversational coherence)
          USER:      retrieved context + question
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
            f"Answer (cite each claim with [N] matching the chunk numbers above):"
        )

        # Build the system prompt: base rules + user customisations + memory.
        # Memory notes are listed as standing facts the model should respect
        # without treating them as retrievable source citations.
        system_parts = [self.GENERATION_SYSTEM_PROMPT]
        if extra_system and extra_system.strip():
            system_parts.append("ADDITIONAL USER INSTRUCTIONS:\n" + extra_system.strip())
        if memory_notes:
            cleaned = [n.strip() for n in memory_notes if n and n.strip()]
            if cleaned:
                bullet_list = "\n".join(f"- {n}" for n in cleaned)
                system_parts.append(
                    "USER MEMORY (standing facts to respect — do NOT cite these as [N] "
                    "since they're not source chunks):\n" + bullet_list
                )
        system_prompt = "\n\n".join(system_parts)

        # Assemble the message list: system prompt, recent history, then the
        # grounded user turn. History gives the model conversational context
        # while the source chunks keep the answer grounded.
        messages: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]
        if chat_history:
            for m in chat_history[-self.HISTORY_TURNS_IN_PROMPT:]:
                role = m.get("role")
                if role in ("user", "assistant") and m.get("content"):
                    messages.append({"role": role, "content": str(m["content"])})
        messages.append({"role": "user", "content": user_message})

        try:
            response = ollama.chat(
                model=self._gen_model,
                messages=messages,
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

    def list_sources(self) -> list[str]:
        """Return all distinct source filenames in the index (for UI filters)."""
        return self._chroma.list_distinct("source")

    def list_modalities(self) -> list[str]:
        """Return all distinct modalities in the index (for UI filters)."""
        return self._chroma.list_distinct("modality")

    def list_sources_detailed(self) -> list[dict]:
        """
        Per-source aggregates for the Knowledge Base tab.

        Returns a list of dicts:
            {"source": str, "modality": str, "chunks": int, "indexed_at": str}

        Sorted by indexed_at descending (newest first). Single pass over Chroma
        metadata — fine for indexes up to ~10⁵ chunks.
        """
        result = self._chroma._collection.get(include=["metadatas"])  # type: ignore[attr-defined]
        agg: dict[str, dict] = {}
        for meta in result.get("metadatas", []) or []:
            src = meta.get("source") or "(unknown)"
            entry = agg.setdefault(src, {
                "source"    : src,
                "modality"  : meta.get("modality", ""),
                "chunks"    : 0,
                "indexed_at": meta.get("indexed_at", ""),
            })
            entry["chunks"] += 1
            # Keep the most recent indexed_at if the source was re-ingested.
            ts = meta.get("indexed_at", "")
            if ts and ts > entry["indexed_at"]:
                entry["indexed_at"] = ts
        return sorted(agg.values(), key=lambda r: r["indexed_at"], reverse=True)

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

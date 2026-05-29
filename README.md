#  Multimodal RAG Assistant

A fully **local**, **free**, end-to-end Retrieval-Augmented Generation system that understands PDFs, Word documents, images, audio, video, and source code — and lets you chat with all of it via a Streamlit UI.

No OpenAI API keys. No cloud services. Everything runs on your machine.

---

##  Features

- **Multimodal ingestion** — PDFs, DOCX, images, MP3/WAV audio, MP4/MKV video, and 20+ code formats
- **Deterministic Markdown chunking** — structure-aware splitter that respects headings, fenced code, tables, and blockquotes; ~200× faster than the legacy LLM chunker and immune to JSON-parse failures on long docs. The LLM chunker is still available via `chunker_strategy="llm"`.
- **Contextual Retrieval** — every chunk is enriched with a situating context sentence before embedding, reducing retrieval failures by ~49% ([Anthropic, 2024](https://www.anthropic.com/news/contextual-retrieval))
- **Hybrid retrieval** — dense vector search (ChromaDB + nomic-embed-text) fused with BM25 sparse search via Reciprocal Rank Fusion (RRF)
- **Optional cross-encoder reranking** — a second-stage `BAAI/bge-reranker-v2-m3` re-scores fused candidates for relevance, typically lifting top-1 precision by ~5–15%
- **Multi-turn conversational retrieval** — follow-up questions with pronouns ("how does that compare to Q2?") are rewritten into self-contained search queries against the chat history before retrieval
- **Cross-file numbered citations** — answers cite each claim with `[1]`, `[2]`, … pointing at a numbered source list, so attribution stays clear even when the response stitches together facts from multiple files
- **Metadata filtering in the UI** — restrict retrieval by **modality** (Document / Audio / Image / Video / Code), **source file**, or **indexed-date range** directly from the sidebar
- **Web URL ingestion** — paste any article URL; `trafilatura` extracts clean main-content Markdown which flows through the same chunking → contextualisation → indexing pipeline as files
- **Persistent memory & prompts** — a user-controlled system prompt + a list of "standing facts" notes that get injected into every answer (no re-indexing needed)
- **Fully local LLMs** — Llama 3 for chunking, context generation, and answering; LLaVA for image understanding; Whisper for transcription
- **Tabbed Next.js + Tailwind UI (v2)** — Chat / Knowledge Base / Memory tabs against a FastAPI backend. Legacy Streamlit UI still works as a fallback.
- **Persistent indexes** — ChromaDB and BM25 survive restarts; re-indexing a file automatically deduplicates

---

## Architecture

```
Uploaded File
     │
     ▼
DataRouter — routes by file extension / MIME type
     │
     ├─► DocumentProcessor  (PDF, DOCX  → Docling → Markdown)
     ├─► AudioProcessor     (MP3, WAV   → Whisper → timestamped transcript)
     ├─► ImageProcessor     (PNG, JPG   → LLaVA   → Markdown description)
     ├─► VideoProcessor     (MP4, MKV   → FFmpeg → audio + keyframes → above)
     └─► CodeProcessor      (PY, JS ... → fenced Markdown code block)
              │
              ▼
     MarkdownUnifier  — prepends YAML metadata header (source, modality, timestamp)
              │
              ▼
     ChunkingAgent (Llama 3)  — semantic chunking → JSON [{chunk_text, summary, keywords}]
              │
              ▼
     ContextualRetriever (Llama 3, parallel)  — adds situating context to each chunk
              │
         ┌───┴───┐
         ▼       ▼
   ChromaDB    BM25Index
  (dense vec)  (sparse kw)
         │       │
         └───┬───┘
             ▼
      HybridRetriever (RRF fusion)
        — sparse-only hits are back-filled from ChromaDB
             │
             ▼
  CrossEncoderReranker (optional, bge-reranker-v2-m3)
        — re-scores candidates by query↔chunk relevance
             │
             ▼
        Llama 3 answer generation
             │
             ▼
       Streamlit Chat UI
```

> The query path over-retrieves (`top_k × 2`) at the fusion stage so the reranker
> has a richer candidate pool to choose from before the final `top_k` is passed to the LLM.

---

##  Quick Start

### 1. Prerequisites

**Python 3.10+** and the following system tools:

| Tool | Purpose | Install |
|------|---------|---------|
| [Ollama](https://ollama.com/download) | Local LLM server | See link |
| FFmpeg | Video/audio processing | `brew install ffmpeg` / `sudo apt install ffmpeg` |

### 2. Pull Ollama Models

```bash
ollama pull llama3            # chunking, context generation, answering
ollama pull llava             # image understanding (~4 GB)
ollama pull nomic-embed-text  # embeddings (~274 MB)

ollama serve                  # keep this running in a terminal
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Whisper and Docling models are downloaded automatically on first use.

**Optional — enable the cross-encoder reranker:**

```bash
pip install sentence-transformers
```

The reranker model (`BAAI/bge-reranker-v2-m3`, ~1.1 GB) downloads automatically the first time you run a query with `use_reranker=True`.

### 4. Run the App (v2 — FastAPI + Next.js)

The v2 UI is a **separate backend + frontend** running in two terminals:

```bash
# Terminal 1 — Python backend (port 8000)
uvicorn api:app --reload --port 8000

# Terminal 2 — Next.js frontend (port 3000)
cd frontend
npm install        # first time only
npm run dev
```

Then open **[http://localhost:3000](http://localhost:3000)**. The frontend
proxies `/api/*` to the Python backend automatically (configured in
`frontend/next.config.js`), so no CORS setup is needed.

> The original Streamlit UI (`streamlit run app.py`) is still in the repo as a
> simpler fallback if you don't want to install Node.

---

## 📖 Usage (v2 UI)

The new UI has three tabs:

| Tab | What it does |
|-----|--------------|
| **Chat** | Conversation with the assistant. Numbered `[1][2]` citations link to the sources panel under each answer. Compact filter bar above the chat lets you scope retrieval by **modality** or **source**. Follow-up questions with pronouns ("how does that compare?") are auto-rewritten using chat history. |
| **Knowledge Base** | Table of every indexed source — filename/title, modality, chunk count, indexed date, delete button. **Upload a file** OR **paste a web URL** to ingest. |
| **Memory & Prompts** | A persistent **custom system prompt** (e.g. "answer concisely in British English") and a list of **memory notes** (standing facts like "Acme is our client since 2019"). Both are injected into every answer automatically — no re-indexing needed. |

### CLI Usage (no UI)

**Ingestion pipeline:**
```bash
# Ingest a file and print chunks to stdout
python multimodal_rag_pipeline.py report.pdf

# Save chunks to JSON
python multimodal_rag_pipeline.py meeting.mp3 chunks.json

# Disable contextual retrieval (v1.0 mode, faster)
python multimodal_rag_pipeline.py diagram.png out.json --no-context
```

**RAG connector:**
```bash
# Index a file
python rag_connector.py index quarterly_report.pdf

# Query the knowledge base
python rag_connector.py query "What were the main revenue drivers in Q3?"

# Query with modality filter
python rag_connector.py query "Summarise the meeting" --modality Audio

# View index stats
python rag_connector.py stats

# Remove a source
python rag_connector.py delete quarterly_report.pdf
```

---

## 📁 Project Structure

```
.
├── api.py                      # FastAPI backend — chat, ingest, sources, memory
├── memory_store.py             # JSON-backed system prompt + memory notes
├── rag_connector.py            # Core façade: index(), index_url(), query()
│   ├── OllamaEmbedder          # nomic-embed-text dense embeddings
│   ├── BM25Index               # Sparse keyword index (bm25_index.json)
│   ├── ChromaStore             # Dense vectors + metadata (./chroma_db/)
│   ├── CrossEncoderReranker    # Optional bge-reranker-v2-m3
│   ├── HybridRetriever         # RRF fusion
│   ├── QueryRewriter           # Multi-turn conversational rewriting
│   └── RAGConnector            # Top-level orchestrator
├── multimodal_rag_pipeline.py  # Ingestion pipeline
│   ├── DataRouter              # File-type dispatcher
│   ├── DocumentProcessor       # PDF/DOCX → Markdown via Docling
│   ├── AudioProcessor          # Audio → transcript via Whisper
│   ├── ImageProcessor          # Image → description via LLaVA
│   ├── VideoProcessor          # Video → audio + keyframes
│   ├── CodeProcessor           # Source code → fenced Markdown
│   ├── URLProcessor            # Web page → clean Markdown via trafilatura
│   ├── MarkdownUnifier         # Attaches metadata header
│   ├── MarkdownChunker         # Deterministic structure-aware splitter (default)
│   ├── ChunkingAgent           # Legacy LLM chunker (opt-in: chunker_strategy="llm")
│   ├── ContextualRetriever     # Parallel context enrichment via Llama 3
│   └── MultimodalRAGPipeline   # High-level orchestrator (ingest + ingest_url)
├── app.py                      # Legacy Streamlit UI (still works)
├── frontend/                   # Next.js 14 + TypeScript + Tailwind UI (v2)
│   ├── app/page.tsx            # 3-tab shell
│   ├── app/components/         # ChatTab · KnowledgeBaseTab · MemoryTab
│   ├── lib/api.ts              # Typed client for the FastAPI backend
│   └── package.json
├── bm25_index.json             # Persisted BM25 corpus (auto-created)
├── chroma_db/                  # ChromaDB storage (auto-created)
├── memory.json                 # User memory state (auto-created)
└── requirements.txt
```

---

## ⚙️ Configuration

All components accept constructor parameters for customisation. Key defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contextual_retrieval` | `True` | Enable/disable Anthropic-style context enrichment |
| `context_max_workers` | `4` | Parallel LLM calls for context generation |
| `embed_model` | `nomic-embed-text` | Ollama embedding model |
| `use_reranker` | `False` | Enable cross-encoder reranking after RRF fusion |
| `reranker_model` | `BAAI/bge-reranker-v2-m3` | sentence-transformers cross-encoder model |
| `generation_model` | `llama3` | Ollama model for final answer generation |
| `keyframe_interval` | `30` | Seconds between video keyframe samples |
| `whisper_model` | `base` | Whisper model size (`tiny`/`base`/`small`/`medium`/`large`) |

Example — enable reranking and use more parallelism for context generation:

```python
from rag_connector import RAGConnector

rag = RAGConnector(
    contextual_retrieval=True,
    context_max_workers=8,   # more parallelism on GPU
    use_reranker=True,       # cross-encoder second-stage ranking
)
rag.index("lecture.mp4")

results = rag.query("What were the key takeaways?", top_k=5)
print(results["answer"])
```

---

## ✂️ How the Markdown Chunker Works

Documents are split into chunks by a deterministic structure-aware walker:

1. **Tokenize** the unified Markdown into structural blocks: headings (h1–h6), paragraphs, fenced code blocks, tables, and blockquotes.
2. **Pack** blocks greedily up to ~1,500 chars per chunk, preferring heading boundaries as split points whenever the buffer is at least half-full.
3. **Preserve atomicity** — code blocks, tables, and blockquotes are never split across chunks (the same rule the LLM chunker tried to follow but occasionally violated).
4. **Long paragraphs** that exceed the max are split at sentence boundaries instead of mid-word.
5. **Outline prepending** — chunks that begin mid-section get the parent heading breadcrumb prepended so they remain self-contained when retrieved out of order. Chunks that already start with a heading need no prefix.
6. **Tiny tail merge** — a final chunk under MIN_CHARS is merged back into its predecessor rather than emitted standalone.

This replaced the prior `ChunkingAgent` (LLM with a JSON-emit prompt) as the default. The LLM chunker silently truncated past `num_ctx=8192` and occasionally returned malformed JSON on long inputs, killing ingest. The deterministic chunker runs in ~100 ms on the same document, never crashes, and emits chunks that are exact substrings of the source (no paraphrasing drift). Both are still available — pick with `MultimodalRAGPipeline(chunker_strategy="markdown" | "llm")`.

---

## 🔬 How Contextual Retrieval Works

Naive chunking strips the surrounding context that makes a chunk interpretable. A chunk reading *"Revenue declined 12% quarter-over-quarter"* is ambiguous in isolation — which company? which quarter?

Contextual Retrieval ([Anthropic, Sept 2024](https://www.anthropic.com/news/contextual-retrieval)) fixes this by passing the **whole document** to the LLM alongside each chunk and asking it to write a 1–2 sentence situating context:

> *"This chunk is from the Acme Corp Q3 2024 earnings report, in the North American retail segment section."*

That context is prepended to the chunk before embedding **and** BM25 indexing. Result: ~49% fewer retrieval failures on Anthropic's benchmarks.

In this project the contextualisation calls run in parallel via `ThreadPoolExecutor`, keeping latency manageable even for large documents.

---

## 🗂️ Metadata Filtering & Cross-File Citations

Every chunk is stored with structured metadata: `source` (filename), `modality`, `summary`, `keywords`, `context`, `indexed_at` (ISO string), and `indexed_ts` (Unix int for range queries).

**Filtering** happens at query time. The sidebar exposes three filters that are combined into a single ChromaDB `where` clause and applied to both retrieval stages:

| Filter | Backend behaviour |
|--------|-------------------|
| **Modality** (multi-select) | `{"modality": {"$in": [...]}}` — dense retrieval filters natively; BM25 results are post-filtered via `ChromaStore.filter_ids` since rank_bm25 is metadata-blind. |
| **Source file** (multi-select) | `{"source": {"$in": [...]}}` — same path. |
| **Indexed date range** | `{"indexed_ts": {"$gte": …, "$lte": …}}` — exact range query over the integer timestamp stored at ingest. |

When more than one constraint is active they are combined with `$and`. Sparse retrieval over-retrieves (`top_k × 4`) when a filter is active so the surviving set is still rich enough for RRF + reranking.

**Citations** were upgraded for multi-file scenarios. The LLM no longer writes `[Source: filename]` inline; instead it cites each claim with `[1]`, `[2]`, … matching a numbered source list rendered in the UI. The system prompt explicitly tells the model to **not blend facts across files** — every numbered claim must come from the chunk with that number. The "Sources used" panel shows how many chunks came from how many distinct files, with a short snippet of each chunk so attribution is verifiable.

---
## 🎯 How Reranking Works

RRF fusion ranks results by their *position* in the dense and sparse result lists — it never directly measures how well a chunk answers the query. This can promote chunks that merely share keywords over chunks that actually contain the answer.

The optional **cross-encoder reranker** fixes this. Unlike the bi-encoder used for embeddings (which encodes query and chunk separately), a cross-encoder feeds the `(query, chunk)` pair through the model *together*, producing a single relevance score that captures their interaction.

```
Query: "What caused the revenue decline?"

  RRF order              →  Reranked order
  1. "Revenue fell 12%"      1. "Cause: market saturation"   ← actually answers it
  2. "Cause: market sat."    2. "Revenue fell 12%"
```

The query pipeline over-retrieves `top_k × 2` candidates at the fusion stage, the reranker scores them all, and only the best `top_k` reach the LLM. Cost is ~100–300 ms for a typical candidate set — negligible next to generation — and it degrades gracefully (falls back to RRF order) if the model fails to load.

Enable it with `RAGConnector(use_reranker=True)` after installing `sentence-transformers`.

---

## Supported File Types

| Category | Extensions |
|----------|-----------|
| Documents | `.pdf`, `.docx`, `.doc`, `.odt`, `.pptx`, `.xlsx` |
| Audio | `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.opus` |
| Images | `.jpg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.webp` |
| Video | `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.m4v` |
| Code | `.py`, `.js`, `.ts`, `.java`, `.c`, `.cpp`, `.go`, `.rs`, `.rb`, `.sh`, `.sql`, `.html`, `.css`, `.yaml`, `.json`, and more |

---

## Dependencies

```
streamlit              # Web UI
ollama                 # Local LLM + embedding inference
chromadb               # Vector database
rank_bm25              # Sparse BM25 retrieval
docling                # PDF / DOCX → Markdown
openai-whisper         # Local speech-to-text
ffmpeg-python          # Video processing bindings
pydantic               # Data validation
sentence-transformers  # Cross-encoder reranker (optional)
```

---

## 🗺️ Roadmap

- [x] Re-ranking with a cross-encoder model
- [x] Multi-turn / conversational retrieval (query rewriting from chat history)
- [x] Multi-document cross-file citation
- [x] Metadata filtering in the UI (by modality, date, source)
- [x] Support for web URL ingestion
- [x] Full UI overhaul (FastAPI + Next.js + Tailwind, 3-tab layout)
- [x] Persistent memory & custom prompts
- [ ] Docker Compose setup for one-command deployment
- [ ] Page-number / timestamp granularity in citations (PDF pages, video timestamps)
- [ ] Streaming token-by-token chat responses

---

## 👤 Author

**Abdelrahman Ezz-Eldin**  
AI Engineering Enthusiast | Computer Engineering Student  
[LinkedIn](https://linkedin.com/in/abdelrahman-ezz-44011a392) · Alexandria, Egypt

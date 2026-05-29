#  Multimodal RAG Assistant

A fully **local**, **free**, end-to-end Retrieval-Augmented Generation system that understands PDFs, Word documents, images, audio, video, and source code — and lets you chat with all of it via a Streamlit UI.

No OpenAI API keys. No cloud services. Everything runs on your machine.

---

##  Features

- **Multimodal ingestion** — PDFs, DOCX, images, MP3/WAV audio, MP4/MKV video, and 20+ code formats
- **Agentic chunking** — LLM-driven semantic segmentation (not naive fixed-size splits)
- **Contextual Retrieval** — every chunk is enriched with a situating context sentence before embedding, reducing retrieval failures by ~49% ([Anthropic, 2024](https://www.anthropic.com/news/contextual-retrieval))
- **Hybrid retrieval** — dense vector search (ChromaDB + nomic-embed-text) fused with BM25 sparse search via Reciprocal Rank Fusion (RRF)
- **Optional cross-encoder reranking** — a second-stage `BAAI/bge-reranker-v2-m3` re-scores fused candidates for relevance, typically lifting top-1 precision by ~5–15%
- **Fully local LLMs** — Llama 3 for chunking, context generation, and answering; LLaVA for image understanding; Whisper for transcription
- **Streamlit chat UI** — chat interface with source attribution and RRF score display
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

### 4. Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📖 Usage

1. **Upload a file** using the sidebar file uploader
2. Click **"Process & Index File"** — the pipeline ingests, chunks, contextualises, and indexes it
3. **Ask questions** in the chat input at the bottom
4. Expand **"📚 View Sources"** under any answer to see which chunks were used and their RRF scores

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
├── app.py                      # Streamlit chat UI
├── rag_connector.py            # Top-level façade: index() and query()
│   ├── OllamaEmbedder          # nomic-embed-text dense embeddings
│   ├── BM25Index               # Sparse keyword index (persisted to bm25_index.json)
│   ├── ChromaStore             # ChromaDB vector store (persisted to ./chroma_db/)
│   ├── CrossEncoderReranker    # Optional second-stage reranking (bge-reranker-v2-m3)
│   ├── HybridRetriever         # RRF fusion of dense + sparse results
│   └── RAGConnector            # Orchestrates all of the above
├── multimodal_rag_pipeline.py  # Ingestion pipeline
│   ├── DataRouter              # File-type dispatcher
│   ├── DocumentProcessor       # PDF/DOCX → Markdown via Docling
│   ├── AudioProcessor          # Audio → transcript via Whisper
│   ├── ImageProcessor          # Image → description via LLaVA
│   ├── VideoProcessor          # Video → audio + keyframes
│   ├── CodeProcessor           # Source code → fenced Markdown
│   ├── MarkdownUnifier         # Attaches metadata header
│   ├── ChunkingAgent           # Agentic semantic chunking via Llama 3
│   ├── ContextualRetriever     # Parallel context enrichment via Llama 3
│   └── MultimodalRAGPipeline   # High-level orchestrator
├── bm25_index.json             # Persisted BM25 corpus (auto-created)
├── chroma_db/                  # ChromaDB storage (auto-created)
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

## 🔬 How Contextual Retrieval Works

Naive chunking strips the surrounding context that makes a chunk interpretable. A chunk reading *"Revenue declined 12% quarter-over-quarter"* is ambiguous in isolation — which company? which quarter?

Contextual Retrieval ([Anthropic, Sept 2024](https://www.anthropic.com/news/contextual-retrieval)) fixes this by passing the **whole document** to the LLM alongside each chunk and asking it to write a 1–2 sentence situating context:

> *"This chunk is from the Acme Corp Q3 2024 earnings report, in the North American retail segment section."*

That context is prepended to the chunk before embedding **and** BM25 indexing. Result: ~49% fewer retrieval failures on Anthropic's benchmarks.

In this project the contextualisation calls run in parallel via `ThreadPoolExecutor`, keeping latency manageable even for large documents.

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
- [ ] Multi-turn / conversational retrieval (query rewriting from chat history)
- [ ] Multi-document cross-file citation
- [ ] Metadata filtering in the UI (by modality, date, source)
- [ ] Support for web URL ingestion
- [ ] Docker Compose setup for one-command deployment

---

## 👤 Author

**Abdelrahman Ezz-Eldin**  
AI Engineering Enthusiast | Computer Engineering Student  
[LinkedIn](https://linkedin.com/in/abdelrahman-ezz-44011a392) · Alexandria, Egypt

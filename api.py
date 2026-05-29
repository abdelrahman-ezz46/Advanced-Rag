"""
================================================================================
  FASTAPI BACKEND — exposes RAGConnector + MemoryStore to the Next.js frontend
================================================================================

Endpoints
---------
  GET  /api/health
  POST /api/chat           — ask a question; supports filters, history, memory
  POST /api/ingest/file    — multipart upload of a file
  POST /api/ingest/url     — ingest a web URL
  GET  /api/sources        — knowledge-base list (file/title, modality, count)
  DELETE /api/sources/{name}
  GET  /api/modalities     — distinct modalities (for the filter dropdown)
  GET  /api/stats          — counts for the header
  GET  /api/memory         — { system_prompt, notes }
  PUT  /api/memory/system  — set the custom system prompt
  POST /api/memory/notes   — add a note
  PUT  /api/memory/notes/{id}
  DELETE /api/memory/notes/{id}

Run:
    uvicorn api:app --reload --port 8000
================================================================================
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from memory_store import MemoryStore
from rag_connector import RAGConnector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# ── Single shared instances ─────────────────────────────────────────────────
# The connector is heavy (loads Ollama/Chroma/BM25); we want exactly one.
USE_RERANKER = os.environ.get("RAG_USE_RERANKER", "false").lower() == "true"
rag    = RAGConnector(use_reranker=USE_RERANKER)
memory = MemoryStore()

# ── App + CORS ──────────────────────────────────────────────────────────────
app = FastAPI(title="Advanced RAG API", version="2.0.0")

# Next.js dev server typically runs on :3000. Allow that origin + localhost
# variants so the frontend can call us cross-port during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    role   : str
    content: str


class ChatRequest(BaseModel):
    question        : str
    top_k           : int               = 5
    chat_history    : list[ChatMessage] = Field(default_factory=list)
    modality_filter : list[str] | None  = None
    source_filter   : list[str] | None  = None
    date_from_ts    : int  | None       = None
    date_to_ts      : int  | None       = None


class URLIngestRequest(BaseModel):
    url: str


class SystemPromptRequest(BaseModel):
    system_prompt: str


class NoteRequest(BaseModel):
    text: str


# ══════════════════════════════════════════════════════════════════════════════
#  HEALTH + STATS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/stats")
def stats() -> dict:
    s = rag.stats
    return {
        "chunks"        : s["chroma_docs"],
        "bm25_chunks"   : s["bm25_docs"],
        "sources"       : len(rag.list_sources()),
        "reranker"      : USE_RERANKER,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    """Run a RAG query with the user's filters + persistent memory injected."""
    try:
        result = rag.query(
            question        = req.question,
            top_k           = req.top_k,
            modality_filter = req.modality_filter or None,
            source_filter   = req.source_filter or None,
            date_from_ts    = req.date_from_ts,
            date_to_ts      = req.date_to_ts,
            chat_history    = [m.model_dump() for m in req.chat_history],
            extra_system    = memory.get_system_prompt() or None,
            memory_notes    = memory.get_note_texts() or None,
        )
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(exc))

    # Trim sources to JSON-safe shape (drop heavy fields if any).
    return {
        "answer"      : result["answer"],
        "search_query": result.get("search_query", req.question),
        "sources"     : [
            {
                "rank"        : s.get("rank"),
                "id"          : s.get("id"),
                "document"    : s.get("document", ""),
                "metadata"    : s.get("metadata", {}),
                "rrf_score"   : s.get("rrf_score"),
                "rerank_score": s.get("rerank_score"),
            }
            for s in result.get("sources", [])
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  INGESTION
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)) -> dict:
    """
    Accept a multipart upload, save it to a temp file with the ORIGINAL suffix
    so the pipeline can route by extension, then run the ingestion pipeline.
    The original filename is preserved as the 'source' in metadata via a rename.
    """
    suffix = Path(file.filename or "").suffix or ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        # Move the temp file next to itself with the user's original filename so
        # the chunks' 'source' metadata reads as "report.pdf", not "tmpXXXX.pdf".
        if file.filename:
            target = tmp_path.parent / file.filename
            try:
                tmp_path.rename(target)
                tmp_path = target
            except OSError:
                pass  # rename across mount points etc — fall back to tmp name

        try:
            n = rag.index(tmp_path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        return {"source": file.filename or tmp_path.name, "chunks": n}
    except Exception as exc:
        logger.exception("File ingest failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ingest/url")
def ingest_url(req: URLIngestRequest) -> dict:
    url = (req.url or "").strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
    try:
        return rag.index_url(url)
    except Exception as exc:
        logger.exception("URL ingest failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE (sources)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/sources")
def list_sources() -> list[dict]:
    return rag.list_sources_detailed()


@app.get("/api/modalities")
def list_modalities() -> list[str]:
    return rag.list_modalities()


@app.delete("/api/sources/{name}")
def delete_source(name: str) -> dict:
    return rag.delete_source(name)


# ══════════════════════════════════════════════════════════════════════════════
#  MEMORY & PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/memory")
def get_memory() -> dict:
    return {
        "system_prompt": memory.get_system_prompt(),
        "notes"        : memory.get_notes(),
    }


@app.put("/api/memory/system")
def set_system_prompt(req: SystemPromptRequest) -> dict:
    memory.set_system_prompt(req.system_prompt)
    return {"system_prompt": memory.get_system_prompt()}


@app.post("/api/memory/notes")
def add_note(req: NoteRequest) -> dict:
    try:
        return memory.add_note(req.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/api/memory/notes/{note_id}")
def update_note(note_id: str, req: NoteRequest) -> dict:
    try:
        updated = memory.update_note(note_id, req.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not updated:
        raise HTTPException(status_code=404, detail="Note not found.")
    return updated


@app.delete("/api/memory/notes/{note_id}")
def delete_note(note_id: str) -> dict:
    ok = memory.delete_note(note_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Note not found.")
    return {"deleted": note_id}

/**
 * Typed client for the FastAPI backend.
 *
 * Uses RELATIVE paths because next.config.js rewrites /api/* → http://localhost:8000/api/*
 * in dev. In production you'd set the backend URL via env and proxy similarly.
 */

// ── Types ─────────────────────────────────────────────────────────────────

export type ChatMessage = { role: "user" | "assistant"; content: string };

export type SourceMeta = {
  source?: string;
  modality?: string;
  summary?: string;
  keywords?: string;
  context?: string;
  indexed_at?: string;
  indexed_ts?: number;
};

export type SourceItem = {
  rank: number;
  id: string;
  document: string;
  metadata: SourceMeta;
  rrf_score?: number;
  rerank_score?: number | null;
};

export type ChatRequest = {
  question: string;
  top_k?: number;
  chat_history?: ChatMessage[];
  modality_filter?: string[] | null;
  source_filter?: string[] | null;
  date_from_ts?: number | null;
  date_to_ts?: number | null;
};

export type ChatResponse = {
  answer: string;
  search_query: string;
  sources: SourceItem[];
};

export type SourceRow = {
  source: string;
  modality: string;
  chunks: number;
  indexed_at: string;
};

export type MemoryNote = { id: string; text: string; created_at: string };
export type MemoryState = { system_prompt: string; notes: MemoryNote[] };
export type Stats = { chunks: number; bm25_chunks: number; sources: number; reranker: boolean };

// ── Helpers ───────────────────────────────────────────────────────────────

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail || JSON.stringify(j);
    } catch {}
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

// ── Endpoints ─────────────────────────────────────────────────────────────

export const api = {
  health  : ()                       => jsonFetch<{ ok: boolean }>("/api/health"),
  stats   : ()                       => jsonFetch<Stats>("/api/stats"),

  chat    : (req: ChatRequest)       => jsonFetch<ChatResponse>("/api/chat", {
                                          method: "POST",
                                          body: JSON.stringify(req),
                                        }),

  ingestUrl: (url: string)           => jsonFetch<{ source: string; chunks: number }>(
                                          "/api/ingest/url",
                                          { method: "POST", body: JSON.stringify({ url }) }
                                        ),

  ingestFile: async (file: File)     => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/ingest/file", { method: "POST", body: fd });
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    return res.json() as Promise<{ source: string; chunks: number }>;
  },

  sources    : ()                    => jsonFetch<SourceRow[]>("/api/sources"),
  modalities : ()                    => jsonFetch<string[]>("/api/modalities"),
  deleteSource: (name: string)       => jsonFetch<{ chroma_deleted: number; bm25_deleted: number }>(
                                          `/api/sources/${encodeURIComponent(name)}`,
                                          { method: "DELETE" }
                                        ),

  memory     : ()                    => jsonFetch<MemoryState>("/api/memory"),
  setSystem  : (system_prompt: string) => jsonFetch<{ system_prompt: string }>(
                                          "/api/memory/system",
                                          { method: "PUT", body: JSON.stringify({ system_prompt }) }
                                        ),
  addNote    : (text: string)        => jsonFetch<MemoryNote>("/api/memory/notes", {
                                          method: "POST", body: JSON.stringify({ text }),
                                        }),
  updateNote : (id: string, text: string) => jsonFetch<MemoryNote>(
                                          `/api/memory/notes/${id}`,
                                          { method: "PUT", body: JSON.stringify({ text }) }
                                        ),
  deleteNote : (id: string)          => jsonFetch<{ deleted: string }>(
                                          `/api/memory/notes/${id}`,
                                          { method: "DELETE" }
                                        ),
};

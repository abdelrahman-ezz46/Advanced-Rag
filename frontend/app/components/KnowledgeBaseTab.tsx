"use client";

import { useEffect, useRef, useState } from "react";
import { Globe, Trash2, Upload, RefreshCw, FileText } from "lucide-react";
import { api, SourceRow } from "../../lib/api";

export default function KnowledgeBaseTab({ onIndexChange }: { onIndexChange?: () => void }) {
  const [rows, setRows]       = useState<SourceRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy]       = useState<string | null>(null);
  const [url, setUrl]         = useState("");
  const [message, setMessage] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function refresh() {
    setLoading(true);
    try {
      setRows(await api.sources());
    } catch (e: any) {
      setMessage({ kind: "err", text: e.message });
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { refresh(); }, []);

  async function handleUpload(file: File) {
    setBusy("upload");
    setMessage(null);
    try {
      const res = await api.ingestFile(file);
      setMessage({ kind: "ok", text: `✓ Indexed ${res.source} (${res.chunks} chunks)` });
      await refresh();
      onIndexChange?.();
    } catch (e: any) {
      setMessage({ kind: "err", text: e.message });
    } finally {
      setBusy(null);
    }
  }

  async function handleIngestUrl() {
    const u = url.trim();
    if (!u) return;
    setBusy("url");
    setMessage(null);
    try {
      const res = await api.ingestUrl(u);
      setMessage({ kind: "ok", text: `✓ Ingested "${res.source}" (${res.chunks} chunks)` });
      setUrl("");
      await refresh();
      onIndexChange?.();
    } catch (e: any) {
      setMessage({ kind: "err", text: e.message });
    } finally {
      setBusy(null);
    }
  }

  async function handleDelete(name: string) {
    if (!confirm(`Delete all chunks from "${name}"?`)) return;
    setBusy(name);
    try {
      await api.deleteSource(name);
      setMessage({ kind: "ok", text: `✓ Deleted ${name}` });
      await refresh();
      onIndexChange?.();
    } catch (e: any) {
      setMessage({ kind: "err", text: e.message });
    } finally {
      setBusy(null);
    }
  }

  const totalChunks = rows.reduce((s, r) => s + r.chunks, 0);

  return (
    <div className="h-full overflow-y-auto px-6 py-6">
      <div className="max-w-5xl mx-auto space-y-6">
        {/* Ingest panel */}
        <div className="grid md:grid-cols-2 gap-4">
          {/* File upload */}
          <div className="bg-panel border border-border rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <Upload size={16} className="text-accent" />
              <h2 className="font-semibold text-sm">Upload a file</h2>
            </div>
            <p className="text-xs text-muted mb-3">
              PDFs, DOCX, images, audio, video, or source code.
            </p>
            <input
              ref={fileInputRef}
              type="file"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleUpload(f);
                e.target.value = "";  // allow re-uploading same file
              }}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={busy === "upload"}
              className="bg-accent text-white text-sm px-4 py-2 rounded-lg hover:bg-accent/90 disabled:opacity-50"
            >
              {busy === "upload" ? "Indexing…" : "Choose file…"}
            </button>
          </div>

          {/* URL ingest */}
          <div className="bg-panel border border-border rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <Globe size={16} className="text-warn" />
              <h2 className="font-semibold text-sm">Ingest a web page</h2>
              <span className="text-[10px] uppercase tracking-wider bg-warn/15 text-warn px-1.5 py-0.5 rounded">new</span>
            </div>
            <p className="text-xs text-muted mb-3">
              Pastes any article URL; we fetch &amp; extract the main content.
            </p>
            <div className="flex gap-2">
              <input
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") handleIngestUrl(); }}
                placeholder="https://example.com/article"
                className="flex-1 bg-panel2 border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-accent"
              />
              <button
                onClick={handleIngestUrl}
                disabled={busy === "url" || !url.trim()}
                className="bg-warn text-white text-sm px-4 py-2 rounded-lg hover:bg-warn/90 disabled:opacity-50"
              >
                {busy === "url" ? "Fetching…" : "Ingest"}
              </button>
            </div>
          </div>
        </div>

        {message && (
          <div
            className={
              "text-sm rounded-lg px-3 py-2 " +
              (message.kind === "ok"
                ? "bg-good/10 text-good border border-good/30"
                : "bg-bad/10 text-bad border border-bad/30")
            }
          >
            {message.text}
          </div>
        )}

        {/* Sources table */}
        <div className="bg-panel border border-border rounded-xl overflow-hidden">
          <div className="flex items-center px-5 py-3 border-b border-border">
            <h2 className="font-semibold text-sm flex items-center gap-2">
              <FileText size={16} /> Indexed sources
            </h2>
            <span className="ml-auto text-xs text-muted">
              {rows.length} source{rows.length === 1 ? "" : "s"} · {totalChunks} chunks
            </span>
            <button
              onClick={refresh}
              className="ml-3 text-muted hover:text-text"
              title="Refresh"
            >
              <RefreshCw size={14} />
            </button>
          </div>

          {loading ? (
            <div className="px-5 py-8 text-center text-muted text-sm">Loading…</div>
          ) : rows.length === 0 ? (
            <div className="px-5 py-12 text-center text-muted text-sm">
              No sources indexed yet — upload a file or paste a URL above.
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead className="text-xs text-muted">
                <tr className="border-b border-border">
                  <th className="text-left font-medium px-5 py-2">Source</th>
                  <th className="text-left font-medium px-3 py-2">Type</th>
                  <th className="text-right font-medium px-3 py-2">Chunks</th>
                  <th className="text-left font-medium px-3 py-2">Indexed</th>
                  <th className="px-3 py-2"></th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.source} className="border-b border-border/60 hover:bg-panel2/40">
                    <td className="px-5 py-2.5 font-mono text-[13px] truncate max-w-[420px]">{r.source}</td>
                    <td className="px-3 py-2.5 text-muted">{r.modality || "—"}</td>
                    <td className="px-3 py-2.5 text-right tabular-nums">{r.chunks}</td>
                    <td className="px-3 py-2.5 text-muted text-xs">{r.indexed_at?.slice(0, 19).replace("T", " ") || "—"}</td>
                    <td className="px-3 py-2.5 text-right">
                      <button
                        onClick={() => handleDelete(r.source)}
                        disabled={busy === r.source}
                        className="text-muted hover:text-bad disabled:opacity-50"
                        title="Delete this source"
                      >
                        <Trash2 size={14} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}

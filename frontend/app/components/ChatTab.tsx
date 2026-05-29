"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChevronDown, ChevronUp, Send, Sparkles, X } from "lucide-react";
import { api, ChatMessage, SourceItem, SourceRow } from "../../lib/api";

// Tiny chip component for the filter bar — keeps the UI compact.
function Chip({ children, onClear }: { children: React.ReactNode; onClear?: () => void }) {
  return (
    <span className="inline-flex items-center gap-1 bg-panel2 border border-border rounded-full px-2.5 py-1 text-xs text-text">
      {children}
      {onClear && (
        <button onClick={onClear} className="text-muted hover:text-bad ml-0.5">
          <X size={12} />
        </button>
      )}
    </span>
  );
}

export default function ChatTab({ onIndexChange }: { onIndexChange?: () => void }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput]       = useState("");
  const [busy, setBusy]         = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState<Record<number, boolean>>({});
  const [lastSources, setLastSources] = useState<SourceItem[] | null>(null);
  const [lastQuery, setLastQuery]     = useState<string>("");

  // Filter state
  const [allSources, setAllSources]       = useState<SourceRow[]>([]);
  const [allModalities, setAllModalities] = useState<string[]>([]);
  const [modalities, setModalities]       = useState<string[]>([]);
  const [sources, setSources]             = useState<string[]>([]);
  const [filtersOpen, setFiltersOpen]     = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.sources().then(setAllSources).catch(() => {});
    api.modalities().then(setAllModalities).catch(() => {});
  }, []);

  // Auto-scroll on new messages
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  const filtersActive = modalities.length + sources.length > 0;

  async function send() {
    const q = input.trim();
    if (!q || busy) return;
    const userMsg: ChatMessage = { role: "user", content: q };
    const next = [...messages, userMsg];
    setMessages(next);
    setInput("");
    setBusy(true);
    try {
      const res = await api.chat({
        question        : q,
        top_k           : 5,
        chat_history    : messages,                       // history WITHOUT current Q
        modality_filter : modalities.length ? modalities : null,
        source_filter   : sources.length ? sources : null,
      });
      setMessages([...next, { role: "assistant", content: res.answer }]);
      setLastSources(res.sources || []);
      setLastQuery(res.search_query || q);
    } catch (e: any) {
      setMessages([...next, { role: "assistant", content: `⚠️ ${e.message}` }]);
      setLastSources(null);
    } finally {
      setBusy(false);
    }
  }

  function clearChat() {
    setMessages([]);
    setLastSources(null);
    setSourcesOpen({});
  }

  return (
    <div className="h-full flex">
      {/* Main column: messages + composer */}
      <div className="flex-1 min-w-0 flex flex-col">
        {/* Filter bar */}
        <div className="border-b border-border bg-panel/70 px-5 py-2 flex items-center gap-2 text-xs flex-wrap">
          <button
            onClick={() => setFiltersOpen((o) => !o)}
            className="text-muted hover:text-text flex items-center gap-1"
          >
            Filters {filtersOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </button>
          {filtersActive && (
            <>
              <span className="text-border">|</span>
              {modalities.map((m) => (
                <Chip key={m} onClear={() => setModalities((xs) => xs.filter((x) => x !== m))}>
                  modality: {m}
                </Chip>
              ))}
              {sources.map((s) => (
                <Chip key={s} onClear={() => setSources((xs) => xs.filter((x) => x !== s))}>
                  source: {s}
                </Chip>
              ))}
              <button
                onClick={() => { setModalities([]); setSources([]); }}
                className="text-muted hover:text-bad ml-1"
              >
                clear all
              </button>
            </>
          )}
          <div className="ml-auto flex items-center gap-3">
            <span className="text-muted">{messages.length / 2 | 0} exchange{messages.length / 2 === 1 ? "" : "s"}</span>
            {messages.length > 0 && (
              <button onClick={clearChat} className="text-muted hover:text-bad">Clear chat</button>
            )}
          </div>
        </div>

        {filtersOpen && (
          <div className="border-b border-border bg-panel/40 px-5 py-3 flex flex-wrap gap-4 text-sm">
            <div className="flex flex-col gap-1.5">
              <label className="text-xs text-muted">Modality</label>
              <select
                multiple
                size={Math.min(4, allModalities.length || 1)}
                value={modalities}
                onChange={(e) => setModalities(Array.from(e.target.selectedOptions).map((o) => o.value))}
                className="bg-panel2 border border-border rounded px-2 py-1 text-text min-w-[160px]"
              >
                {allModalities.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
            <div className="flex flex-col gap-1.5">
              <label className="text-xs text-muted">Source</label>
              <select
                multiple
                size={Math.min(4, allSources.length || 1)}
                value={sources}
                onChange={(e) => setSources(Array.from(e.target.selectedOptions).map((o) => o.value))}
                className="bg-panel2 border border-border rounded px-2 py-1 text-text min-w-[260px]"
              >
                {allSources.map((s) => (
                  <option key={s.source} value={s.source}>
                    {s.source} ({s.chunks})
                  </option>
                ))}
              </select>
            </div>
            <p className="text-xs text-muted self-end">
              Hold ⌘/Ctrl to select multiple. Empty = no filter.
            </p>
          </div>
        )}

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-5 py-5 space-y-4">
          {messages.length === 0 && (
            <div className="text-center text-muted mt-16">
              <Sparkles className="inline mr-2" size={16} />
              Ask a question about your indexed documents.
            </div>
          )}
          {messages.map((m, i) => {
            const isAssistant = m.role === "assistant";
            const isLastAssistant = isAssistant && i === messages.length - 1 && lastSources;
            return (
              <div key={i} className={"flex " + (isAssistant ? "justify-start" : "justify-end")}>
                <div
                  className={
                    "max-w-[78%] rounded-2xl px-4 py-2.5 text-[14.5px] " +
                    (isAssistant
                      ? "bg-panel border border-border text-text"
                      : "bg-accent/90 text-white")
                  }
                >
                  <div className="prose-chat">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                  </div>
                  {isLastAssistant && lastSources!.length > 0 && (
                    <SourcesPanel
                      open={sourcesOpen[i] ?? true}
                      onToggle={() => setSourcesOpen((s) => ({ ...s, [i]: !(s[i] ?? true) }))}
                      sources={lastSources!}
                      rewritten={lastQuery !== messages[i - 1]?.content ? lastQuery : ""}
                    />
                  )}
                </div>
              </div>
            );
          })}
          {busy && (
            <div className="flex justify-start">
              <div className="bg-panel border border-border rounded-2xl px-4 py-2.5 text-sm text-muted">
                <span className="inline-block animate-pulse">Thinking…</span>
              </div>
            </div>
          )}
        </div>

        {/* Composer */}
        <div className="border-t border-border bg-panel px-5 py-3">
          <div className="flex gap-2 items-end">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
              }}
              rows={1}
              placeholder="Ask a question…   (Enter to send · Shift+Enter for newline)"
              className="flex-1 bg-panel2 border border-border rounded-lg px-3 py-2.5 text-sm resize-none focus:outline-none focus:border-accent"
            />
            <button
              onClick={send}
              disabled={busy || !input.trim()}
              className="bg-accent text-white rounded-lg px-4 py-2.5 text-sm font-medium hover:bg-accent/90 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5"
            >
              <Send size={14} /> Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Inline sources panel under the last assistant message.
function SourcesPanel({
  open, onToggle, sources, rewritten,
}: {
  open: boolean;
  onToggle: () => void;
  sources: SourceItem[];
  rewritten: string;
}) {
  const uniqueFiles = useMemo(
    () => new Set(sources.map((s) => s.metadata?.source || "?")),
    [sources]
  );
  return (
    <div className="mt-3 pt-3 border-t border-border">
      <button
        onClick={onToggle}
        className="text-xs text-muted hover:text-text flex items-center gap-1"
      >
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        Sources ({sources.length} chunks across {uniqueFiles.size} file{uniqueFiles.size === 1 ? "" : "s"})
      </button>
      {open && (
        <div className="mt-2 space-y-2">
          {rewritten && (
            <div className="text-xs text-muted italic">↻ Rewritten query: {rewritten}</div>
          )}
          {sources.map((s, idx) => {
            const snippet = (s.document || "").replace(/\s+/g, " ").trim().slice(0, 320);
            return (
              <div key={s.id || idx} className="text-xs bg-panel2/70 border border-border rounded-lg p-2.5">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-semibold text-text">[{s.rank}]</span>
                  <span className="font-mono text-accent">{s.metadata?.source || "?"}</span>
                  <span className="text-muted">· {s.metadata?.modality || "?"}</span>
                  {s.rrf_score !== undefined && (
                    <span className="text-muted">· RRF {s.rrf_score!.toFixed(3)}</span>
                  )}
                  {s.rerank_score != null && (
                    <span className="text-muted">· Rerank {s.rerank_score.toFixed(3)}</span>
                  )}
                </div>
                {snippet && (
                  <div className="text-muted mt-1.5 leading-snug">
                    {snippet}{(s.document?.length || 0) > 320 ? "…" : ""}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

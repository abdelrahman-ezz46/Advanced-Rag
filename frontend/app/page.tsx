"use client";

import { useEffect, useState } from "react";
import { Brain, MessageSquare, Database, Cog } from "lucide-react";
import { api, Stats } from "../lib/api";
import ChatTab from "./components/ChatTab";
import KnowledgeBaseTab from "./components/KnowledgeBaseTab";
import MemoryTab from "./components/MemoryTab";

type TabKey = "chat" | "kb" | "memory";

const TABS: { key: TabKey; label: string; icon: React.ReactNode }[] = [
  { key: "chat",   label: "Chat",            icon: <MessageSquare size={16} /> },
  { key: "kb",     label: "Knowledge Base",  icon: <Database     size={16} /> },
  { key: "memory", label: "Memory & Prompts", icon: <Cog         size={16} /> },
];

export default function Home() {
  const [tab, setTab] = useState<TabKey>("chat");
  const [stats, setStats] = useState<Stats | null>(null);

  // Reload stats whenever the tab changes — cheap, keeps the header honest
  // about how many chunks / sources are currently indexed.
  useEffect(() => {
    api.stats().then(setStats).catch(() => setStats(null));
  }, [tab]);

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-border bg-panel px-5 py-3 flex items-center gap-4">
        <div className="flex items-center gap-2">
          <Brain className="text-accent" size={22} />
          <span className="font-semibold text-[15px]">Advanced RAG</span>
          <span className="text-muted text-xs">· local multimodal assistant</span>
        </div>
        <div className="ml-auto flex items-center gap-4 text-xs text-muted">
          {stats ? (
            <>
              <span><span className="text-text">{stats.chunks}</span> chunks</span>
              <span><span className="text-text">{stats.sources}</span> sources</span>
              <span>reranker: {stats.reranker ? <span className="text-good">on</span> : <span className="text-muted">off</span>}</span>
            </>
          ) : (
            <span className="text-bad">backend offline</span>
          )}
        </div>
      </header>

      {/* Tabs */}
      <nav className="border-b border-border bg-panel px-5 flex gap-1">
        {TABS.map((t) => {
          const active = tab === t.key;
          return (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={
                "px-4 py-2.5 text-sm flex items-center gap-2 border-b-2 -mb-px transition-colors " +
                (active
                  ? "border-accent text-text"
                  : "border-transparent text-muted hover:text-text")
              }
            >
              {t.icon}
              {t.label}
            </button>
          );
        })}
      </nav>

      {/* Tab body */}
      <main className="flex-1 min-h-0 overflow-hidden">
        {tab === "chat"   && <ChatTab onIndexChange={() => api.stats().then(setStats)} />}
        {tab === "kb"     && <KnowledgeBaseTab onIndexChange={() => api.stats().then(setStats)} />}
        {tab === "memory" && <MemoryTab />}
      </main>
    </div>
  );
}

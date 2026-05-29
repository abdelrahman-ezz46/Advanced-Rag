"use client";

import { useEffect, useState } from "react";
import { Check, Edit3, Plus, Save, Trash2, X } from "lucide-react";
import { api, MemoryNote, MemoryState } from "../../lib/api";

export default function MemoryTab() {
  const [state, setState]               = useState<MemoryState | null>(null);
  const [systemDraft, setSystemDraft]   = useState("");
  const [systemDirty, setSystemDirty]   = useState(false);
  const [systemSaving, setSystemSaving] = useState(false);
  const [newNote, setNewNote]           = useState("");
  const [editingId, setEditingId]       = useState<string | null>(null);
  const [editText, setEditText]         = useState("");
  const [msg, setMsg]                   = useState<{ kind: "ok" | "err"; text: string } | null>(null);

  async function refresh() {
    const m = await api.memory();
    setState(m);
    setSystemDraft(m.system_prompt);
    setSystemDirty(false);
  }
  useEffect(() => { refresh().catch(() => {}); }, []);

  async function saveSystem() {
    setSystemSaving(true);
    setMsg(null);
    try {
      await api.setSystem(systemDraft);
      setSystemDirty(false);
      setMsg({ kind: "ok", text: "✓ System prompt saved." });
    } catch (e: any) {
      setMsg({ kind: "err", text: e.message });
    } finally {
      setSystemSaving(false);
    }
  }

  async function addNote() {
    const t = newNote.trim();
    if (!t) return;
    try {
      await api.addNote(t);
      setNewNote("");
      await refresh();
    } catch (e: any) {
      setMsg({ kind: "err", text: e.message });
    }
  }

  async function saveEdit(id: string) {
    const t = editText.trim();
    if (!t) return;
    try {
      await api.updateNote(id, t);
      setEditingId(null);
      await refresh();
    } catch (e: any) {
      setMsg({ kind: "err", text: e.message });
    }
  }

  async function deleteNote(id: string) {
    if (!confirm("Delete this memory note?")) return;
    await api.deleteNote(id);
    await refresh();
  }

  return (
    <div className="h-full overflow-y-auto px-6 py-6">
      <div className="max-w-3xl mx-auto space-y-6">
        {/* Intro */}
        <div className="text-sm text-muted">
          These settings travel with every question you ask — they let the model behave
          consistently and remember standing facts <em>without</em> re-indexing any documents.
        </div>

        {/* System prompt */}
        <section className="bg-panel border border-border rounded-xl p-5">
          <header className="flex items-center mb-3">
            <h2 className="font-semibold text-sm">Custom system instructions</h2>
            <span className="ml-auto text-xs text-muted">
              appended to the assistant's base system prompt
            </span>
          </header>
          <textarea
            value={systemDraft}
            onChange={(e) => { setSystemDraft(e.target.value); setSystemDirty(true); }}
            rows={5}
            placeholder="e.g. Answer concisely. Use British English. Always define acronyms on first use."
            className="w-full bg-panel2 border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-accent resize-y"
          />
          <div className="flex items-center mt-3">
            <span className="text-xs text-muted">
              {systemDirty ? "Unsaved changes" : (systemDraft ? "Active." : "Empty.")}
            </span>
            <button
              onClick={saveSystem}
              disabled={!systemDirty || systemSaving}
              className="ml-auto bg-accent text-white text-sm px-4 py-1.5 rounded-lg hover:bg-accent/90 disabled:opacity-40 flex items-center gap-1.5"
            >
              <Save size={13} /> {systemSaving ? "Saving…" : "Save"}
            </button>
          </div>
        </section>

        {/* Memory notes */}
        <section className="bg-panel border border-border rounded-xl p-5">
          <header className="flex items-center mb-3">
            <h2 className="font-semibold text-sm">Memory notes</h2>
            <span className="ml-auto text-xs text-muted">
              injected as standing facts (not cited as [N])
            </span>
          </header>

          <div className="flex gap-2 mb-4">
            <input
              value={newNote}
              onChange={(e) => setNewNote(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") addNote(); }}
              placeholder="e.g. Acme has been our client since 2019."
              className="flex-1 bg-panel2 border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-accent"
            />
            <button
              onClick={addNote}
              disabled={!newNote.trim()}
              className="bg-accent2 text-white text-sm px-4 py-2 rounded-lg hover:bg-accent2/90 disabled:opacity-40 flex items-center gap-1.5"
            >
              <Plus size={14} /> Add
            </button>
          </div>

          {!state || state.notes.length === 0 ? (
            <div className="text-sm text-muted text-center py-6">
              No memory notes yet. Add a fact and it will be available on every future answer.
            </div>
          ) : (
            <ul className="space-y-2">
              {state.notes.map((n) => (
                <li
                  key={n.id}
                  className="bg-panel2/60 border border-border rounded-lg px-3 py-2.5 flex items-center gap-3"
                >
                  {editingId === n.id ? (
                    <>
                      <input
                        value={editText}
                        onChange={(e) => setEditText(e.target.value)}
                        onKeyDown={(e) => { if (e.key === "Enter") saveEdit(n.id); }}
                        autoFocus
                        className="flex-1 bg-bg border border-border rounded px-2 py-1 text-sm focus:outline-none focus:border-accent"
                      />
                      <button onClick={() => saveEdit(n.id)} className="text-good hover:opacity-80"><Check size={16} /></button>
                      <button onClick={() => setEditingId(null)} className="text-muted hover:text-bad"><X size={16} /></button>
                    </>
                  ) : (
                    <>
                      <span className="flex-1 text-sm">{n.text}</span>
                      <span className="text-[10px] text-muted whitespace-nowrap">
                        {n.created_at?.slice(0, 10)}
                      </span>
                      <button
                        onClick={() => { setEditingId(n.id); setEditText(n.text); }}
                        className="text-muted hover:text-text"
                        title="Edit"
                      >
                        <Edit3 size={14} />
                      </button>
                      <button
                        onClick={() => deleteNote(n.id)}
                        className="text-muted hover:text-bad"
                        title="Delete"
                      >
                        <Trash2 size={14} />
                      </button>
                    </>
                  )}
                </li>
              ))}
            </ul>
          )}
        </section>

        {msg && (
          <div
            className={
              "text-sm rounded-lg px-3 py-2 " +
              (msg.kind === "ok"
                ? "bg-good/10 text-good border border-good/30"
                : "bg-bad/10 text-bad border border-bad/30")
            }
          >
            {msg.text}
          </div>
        )}
      </div>
    </div>
  );
}

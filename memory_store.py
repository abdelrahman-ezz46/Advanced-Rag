"""
================================================================================
  MEMORY STORE — Persistent system prompt + user memory notes
================================================================================

Lightweight JSON-backed store for the "Memory & Prompts" tab.

Two user-controlled pieces of state are persisted between sessions:

  1. system_prompt : str
        Free-form instructions appended to the base generation system prompt.
        Example: "Answer in British English. Always define acronyms on first use."

  2. notes : list[dict]
        Standing facts the user wants the model to respect on every query.
        Each entry: {"id": str, "text": str, "created_at": ISO string}.
        Example: "Acme has been our client since 2019."

Both are injected into the LLM's system prompt by RAGConnector._generate(),
so they take effect immediately on the next question — no re-indexing required.

The file is human-readable JSON so you can hand-edit it if you ever need to.
================================================================================
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("MemoryStore")


class MemoryStore:
    """JSON-backed store for the user's system prompt and memory notes."""

    DEFAULT_PATH = "memory.json"

    def __init__(self, persist_path: str | Path = DEFAULT_PATH) -> None:
        self._path: Path = Path(persist_path)
        self._data: dict = {"system_prompt": "", "notes": []}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                # Validate shape; fall back to defaults on corruption.
                if isinstance(raw, dict):
                    self._data["system_prompt"] = str(raw.get("system_prompt", ""))
                    notes = raw.get("notes", [])
                    if isinstance(notes, list):
                        self._data["notes"] = [n for n in notes if isinstance(n, dict)]
                logger.info(
                    "[MemoryStore] Loaded — system_prompt: %d chars, notes: %d",
                    len(self._data["system_prompt"]),
                    len(self._data["notes"]),
                )
            except Exception as exc:
                logger.warning("[MemoryStore] Could not load — starting fresh (%s).", exc)

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── System prompt ────────────────────────────────────────────────────────

    def get_system_prompt(self) -> str:
        return self._data.get("system_prompt", "")

    def set_system_prompt(self, text: str) -> None:
        self._data["system_prompt"] = str(text or "")
        self._save()

    # ── Notes ────────────────────────────────────────────────────────────────

    def get_notes(self) -> list[dict]:
        """Return all notes as a list of dicts (id, text, created_at)."""
        return list(self._data.get("notes", []))

    def get_note_texts(self) -> list[str]:
        """Return just the note bodies — used to inject into the LLM prompt."""
        return [n.get("text", "") for n in self._data.get("notes", []) if n.get("text")]

    def add_note(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            raise ValueError("Note text cannot be empty.")
        note = {
            "id"        : str(uuid.uuid4()),
            "text"      : text,
            "created_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        }
        self._data["notes"].append(note)
        self._save()
        return note

    def update_note(self, note_id: str, text: str) -> dict | None:
        text = (text or "").strip()
        if not text:
            raise ValueError("Note text cannot be empty.")
        for n in self._data["notes"]:
            if n.get("id") == note_id:
                n["text"] = text
                self._save()
                return n
        return None

    def delete_note(self, note_id: str) -> bool:
        before = len(self._data["notes"])
        self._data["notes"] = [n for n in self._data["notes"] if n.get("id") != note_id]
        if len(self._data["notes"]) < before:
            self._save()
            return True
        return False

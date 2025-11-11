"""
memory.py
---------
Memory & checkpointing utilities.

Newer `langgraph-checkpoint-sqlite` exposes SqliteSaver.from_conn_string(...)
as a *context manager*. We must ENTER it and keep it alive for the app
lifetime. This helper does exactly that, and falls back to MemorySaver
if the SQLite saver isn't available.

It also provides a simple per-session JSON KV store for long-term memory.
"""
from __future__ import annotations

import atexit
import json
from pathlib import Path
from typing import Any, Dict

# Optional SQLite saver
_HAS_SQLITE = False
SqliteSaver = None  # type: ignore[attr-defined]

try:
    # pip install langgraph-checkpoint-sqlite
    from langgraph.checkpoint.sqlite import SqliteSaver as _SqliteSaver  # type: ignore
    SqliteSaver = _SqliteSaver
    _HAS_SQLITE = True
except Exception:
    pass

# Always available fallback
try:
    from langgraph.checkpoint.memory import MemorySaver
except Exception as e:
    raise RuntimeError(
        "LangGraph missing MemorySaver. Ensure `langgraph>=0.2.x` is installed."
    ) from e


# Keep a single entered context for the process lifetime
_entered_ctx = None  # the context manager object
_entered_saver = None  # the actual checkpointer (has get_next_version, etc.)


def _enter_sqlite_ctx(conn_str: str):
    """
    Enter the SqliteSaver context once and keep it open.
    Registers an atexit hook to close cleanly.
    """
    global _entered_ctx, _entered_saver
    if _entered_saver is not None:
        return _entered_saver

    # Ensure directory exists
    Path(conn_str).parent.mkdir(parents=True, exist_ok=True)

    # Enter the context manager
    _entered_ctx = SqliteSaver.from_conn_string(conn_str)  # returns a context manager
    _entered_saver = _entered_ctx.__enter__()              # actual saver object

    # Close on interpreter exit
    def _close():
        try:
            if _entered_ctx is not None:
                _entered_ctx.__exit__(None, None, None)
        except Exception:
            pass

    atexit.register(_close)
    return _entered_saver


def make_checkpointer(sqlite_path: str):
    """
    Return a LangGraph checkpointer object (not a context manager).
    Prefers persistent SQLite; falls back to in-memory if unavailable.
    """
    if _HAS_SQLITE and SqliteSaver is not None:
        try:
            return _enter_sqlite_ctx(sqlite_path)
        except Exception:
            # Fall through to MemorySaver if there’s any issue
            pass
    return MemorySaver()


class SessionKVStore:
    """
    Minimal session-scoped KV store (JSON-backed) to persist long-term nuggets
    (e.g., user's preferences or prior conclusions). Not required by LangGraph,
    but useful for continuity beyond the checkpoint lifetime.
    """
    def __init__(self, persist_dir: str) -> None:
        self.root = Path(persist_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def read(self, session_id: str) -> Dict[str, Any]:
        p = self._path(session_id)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def write(self, session_id: str, data: Dict[str, Any]) -> None:
        p = self._path(session_id)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

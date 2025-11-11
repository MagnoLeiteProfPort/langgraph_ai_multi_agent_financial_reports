"""
nodes.py
--------
Agent node implementations for the LangGraph pipeline.

This module defines:
- A tool registry and a safe `call_tool` wrapper
- The Researcher agent step (can propose tool calls via a JSON block)
- The Reviewer agent step (self-critique / refinement)

Key design notes:
- We import message classes from `langchain_core.messages` (compatible with LangChain >= 0.3).
- We accept history as a list of dicts (role/content) for the LLM prompt. The graph itself
  stores `Message` Pydantic objects; graph.py normalizes to dicts before calling us, and we
  defensively normalize again here.
- Optional offline/dev mode (`DEV_NO_LLM=true`) bypasses LLM calls so you can smoke-test the app
  without OpenAI credentials.
"""

from __future__ import annotations

import os
import re
import json
from typing import Dict, Any, List, Tuple

from langchain_openai import ChatOpenAI
# New import location in modern LangChain:
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .prompts import SYSTEM_RESEARCHER, SYSTEM_REVIEWER
from .tools.finance_tools import get_price, key_metrics
from .tools.web_tools import search_news
from .tools.critique import checklist


# --------------------------------------------------------------------------------------
# Configuration: enable a dev/offline mode that bypasses LLM calls for smoke testing.
# Set DEV_NO_LLM=true (or 1/yes) in your environment to enable.
# --------------------------------------------------------------------------------------
DEV_NO_LLM = os.getenv("DEV_NO_LLM", "").strip().lower() in {"1", "true", "yes"}


# --------------------------------------------------------------------------------------
# Tool registry & safe dispatcher
# --------------------------------------------------------------------------------------
TOOLS: Dict[str, Any] = {
    "get_price": get_price,
    "key_metrics": key_metrics,
    "search_news": search_news,
}

def call_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """
    Safely dispatch a tool call by name. Returns a JSON-serializable dict.
    Any exception is caught and returned as {"error": "..."} so the graph continues.
    """
    fn = TOOLS.get(tool_name)
    if not fn:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        return fn(**kwargs)
    except Exception as e:
        return {"error": str(e)}


# --------------------------------------------------------------------------------------
# Offline stubs (used when DEV_NO_LLM is true or OPENAI key is missing)
# --------------------------------------------------------------------------------------
def _offline_stub_research(question: str, last_obs: Dict[str, Any] | None) -> str:
    """
    Produce a deterministic placeholder draft to validate end-to-end plumbing without LLMs.
    """
    obs_text = f"\n\nRecent tool observation: {last_obs}" if last_obs else ""
    return (
        "# Draft (offline)\n\n"
        f"Task: {question}{obs_text}\n\n"
        "- Key drivers: [placeholder]\n"
        "- Catalysts: [placeholder]\n"
        "- Risks: [placeholder]\n\n"
        "TOOL_CALLS: []"
    )

def _offline_stub_review(draft: str) -> str:
    """
    Provide a deterministic 'reviewed' version in offline mode.
    """
    notes = checklist(draft)
    bullet = "\n- ".join(notes) if notes else "No additional notes."
    return draft + f"\n\n(Reviewed offline)\n- {bullet}"


# --------------------------------------------------------------------------------------
# Researcher agent
# --------------------------------------------------------------------------------------
def researcher_step(
    api_key: str,
    question: str,
    history: List[Dict[str, Any]] | List[Any],
    last_obs: Dict[str, Any] | None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Produce a new draft and optional tool calls using a ReAct-style prompt.

    Parameters
    ----------
    api_key : str
        OpenAI API key (empty string allowed; dev mode may bypass LLM).
    question : str
        The latest user request to address.
    history : list
        Recent conversation history. Expected shape: list of dicts with keys:
        {"role": "user"|"assistant"|"system"|"tool", "content": str}
        We defensively normalize input in case Pydantic Message objects slip through.
    last_obs : dict | None
        Most recent tool result (if any), appended as context to the prompt.

    Returns
    -------
    (draft_text, tool_calls)
        draft_text : str
            The Researcher's drafted answer (may include a TOOL_CALLS JSON block).
        tool_calls : list[dict]
            Parsed list of proposed tool calls if 'TOOL_CALLS: [...]' is found; else [].
            Each item should look like {"tool": "get_price", "args": {"symbol": "NVDA"}}
    """
    # --- Normalize history to list[dict] defensively ---
    norm_history: List[Dict[str, Any]] = []
    for h in (history or []):
        try:
            if hasattr(h, "model_dump"):
                norm_history.append(h.model_dump())
            elif isinstance(h, dict):
                norm_history.append({"role": h.get("role", ""), "content": h.get("content", "")})
            else:
                # Best-effort conversion
                norm_history.append(dict(h))  # may raise; caught by outer try
        except Exception:
            # Ignore malformed history entries
            continue

    # --- Offline/dev path (no LLM) ---
    if DEV_NO_LLM or not api_key:
        return _offline_stub_research(question, last_obs), []

    # --- Online path (LLM) ---
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)

    msgs: List[Any] = [SystemMessage(content=SYSTEM_RESEARCHER)]
    # Use the last few turns to keep prompt lean
    for m in norm_history[-6:]:
        role = (m.get("role") or "").strip().lower()
        content = m.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
        elif role == "system":
            msgs.append(SystemMessage(content=content))
        elif role == "tool":
            # Model can't "see" structured tool messages natively; include as plain text context.
            msgs.append(HumanMessage(content=f"(Tool observation) {content}"))
        else:
            # Unknown role → include as user text to avoid losing context
            msgs.append(HumanMessage(content=content))

    if last_obs:
        msgs.append(HumanMessage(content=f"Recent tool observation: {last_obs}"))

    msgs.append(
        HumanMessage(
            content=(
                f"Task: {question}\n\n"
                "If you need external data, propose tool calls as a JSON list labeled exactly:\n"
                "TOOL_CALLS: [\n"
                '  {"tool": "get_price", "args": {"symbol": "NVDA"}},\n'
                '  {"tool": "search_news", "args": {"query": "NVIDIA earnings"}}\n'
                "]\n"
                "Otherwise, write your draft directly. Prefer bullets for findings."
            )
        )
    )

    out = llm.invoke(msgs).content
    tool_calls: List[Dict[str, Any]] = []

    # --- Extract TOOL_CALLS JSON block if present ---
    m = re.search(r"TOOL_CALLS\s*:\s*(\[[\s\S]*?\])", out, re.IGNORECASE)
    if m:
        try:
            # Be resilient to trailing commas / minor JSON issues
            block = m.group(1)
            tool_calls = json.loads(block)
            # Normalize items to {tool, args}
            cleaned: List[Dict[str, Any]] = []
            for item in tool_calls:
                if not isinstance(item, dict):
                    continue
                name = item.get("tool") or item.get("tool_name")
                args = item.get("args", {})
                if name and isinstance(args, dict):
                    cleaned.append({"tool": name, "args": args})
            tool_calls = cleaned
        except Exception:
            tool_calls = []

    return out, tool_calls


# --------------------------------------------------------------------------------------
# Reviewer agent (self-critique)
# --------------------------------------------------------------------------------------
def reviewer_step(api_key: str, draft: str) -> str:
    """
    Critically review and improve the Researcher draft.

    Behavior:
    - Applies a simple checklist (from tools/critique.py) to enforce sections like Risks/Catalysts.
    - In dev/offline mode, returns a deterministic annotated draft without calling an LLM.
    """
    # --- Offline/dev path (no LLM) ---
    if DEV_NO_LLM or not api_key:
        return _offline_stub_review(draft)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)
    notes = checklist(draft)
    checklist_text = "- " + "\n- ".join(notes) if notes else "- No additional notes."
    prompt = (
        f"{SYSTEM_REVIEWER}\n\n"
        f"DRAFT:\n{draft}\n\n"
        f"CHECKLIST:\n{checklist_text}\n\n"
        "Rewrite the draft, applying the checklist verbatim, preserving factual claims unless flagged."
    )
    return llm.invoke([HumanMessage(content=prompt)]).content

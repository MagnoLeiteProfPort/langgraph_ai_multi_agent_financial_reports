"""
graph.py
--------
LangGraph wiring. Defines the state machine and node transitions.

Key fixes in this version:
- Robust handling of the graph's return type (dict OR Pydantic model).
- History is stored as `Message` objects, but node steps normalize to dicts
  before calling LLM helpers.
- Safe extraction of fields from the final result regardless of type.

Flow:
START -> research -> (tool loop/research) -> review -> guard -> END
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple

from langgraph.graph import StateGraph, START, END

from .guardrails import validate_output
from ..models import GraphState, Message
from .nodes import researcher_step, reviewer_step, call_tool
from .memory import SessionKVStore
from ..config import settings


MAX_STEPS = 3


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _append(state: GraphState, role: str, content: str) -> None:
    """
    Append a message to the state's history as a `Message` object.
    """
    state.history.append(Message(role=role, content=content))


def _to_msg_dicts(history: List[Message]) -> List[Dict[str, Any]]:
    """
    Convert a list of Message objects to a list of plain dicts.
    """
    out: List[Dict[str, Any]] = []
    for m in history:
        try:
            out.append(m.model_dump())
        except Exception:
            out.append({"role": getattr(m, "role", ""), "content": getattr(m, "content", "")})
    return out


def _get_last_by_role(history: List[Message], role: str) -> str:
    """
    Find the content of the most recent message with the given role.
    """
    for m in reversed(history):
        if getattr(m, "role", None) == role:
            return getattr(m, "content", "") or ""
    return ""


def _result_to_dict(result: Any) -> Dict[str, Any]:
    """
    Normalize a LangGraph invoke result (which may be a Pydantic model or a dict)
    into a plain dict for consistent field access.
    """
    if result is None:
        return {}
    # Pydantic v2 models
    if hasattr(result, "model_dump"):
        try:
            return result.model_dump()
        except Exception:
            pass
    # Pydantic v1 fallback
    if hasattr(result, "dict"):
        try:
            return result.dict()
        except Exception:
            pass
    # Already a dict?
    if isinstance(result, dict):
        return result
    # Last resort: try __dict__
    try:
        return dict(result)  # may raise
    except Exception:
        try:
            return vars(result)  # may raise
        except Exception:
            return {}


def _history_to_dicts(history_any: Any) -> List[Dict[str, Any]]:
    """
    Convert an arbitrary `history` (list of Message objects or dicts) into list[dict].
    """
    out: List[Dict[str, Any]] = []
    for h in history_any or []:
        if hasattr(h, "model_dump"):
            try:
                out.append(h.model_dump())
                continue
            except Exception:
                pass
        if isinstance(h, dict):
            out.append({"role": h.get("role", ""), "content": h.get("content", "")})
            continue
        # Best-effort conversion
        try:
            out.append(dict(h))
        except Exception:
            out.append({"role": getattr(h, "role", ""), "content": getattr(h, "content", "")})
    return out


# --------------------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------------------
def node_research(state: GraphState) -> GraphState:
    """
    Researcher step: produce a draft and optionally enqueue tool calls.
    """
    question = _get_last_by_role(state.history, "user")
    history_dicts = _to_msg_dicts(state.history)

    draft, tool_calls = researcher_step(
        settings.openai_api_key,
        question,
        history_dicts,
        state.last_tool_result,
    )
    _append(state, "assistant", draft)
    state.scratch["tool_queue"] = tool_calls
    state.steps += 1
    return state


def node_tool(state: GraphState) -> GraphState:
    """
    Execute exactly one pending tool call from the queue (if any).
    """
    queue: List[Dict[str, Any]] = state.scratch.get("tool_queue", [])
    if not queue:
        return state

    call = queue.pop(0)
    name = call.get("tool") or call.get("tool_name")
    args = call.get("args", {}) if isinstance(call.get("args"), dict) else {}

    result = call_tool(name, **args) if name else {"error": "Invalid tool call schema."}
    state.last_tool_result = result
    _append(state, "tool", f"{name}: {result}")
    state.scratch["tool_queue"] = queue
    return state


def node_review(state: GraphState) -> GraphState:
    """
    Reviewer step: critique and improve the latest assistant draft.
    """
    last_assistant = _get_last_by_role(state.history, "assistant")
    improved = reviewer_step(settings.openai_api_key, last_assistant or "Empty draft.")
    _append(state, "assistant", improved)
    return state


def node_guard(state: GraphState) -> GraphState:
    """
    Guardrails step: validate output and add flags/disclaimer as needed.
    """
    last_assistant = _get_last_by_role(state.history, "assistant")
    safe_text, flags = validate_output(last_assistant)
    state.guardrail_flags.extend(flags)
    state.output = safe_text
    return state


def router(state: GraphState) -> str:
    """
    Route transitions:
    - If there are pending tool calls, go to 'tool'.
    - If we've just executed a tool (last_tool_result is set) and max steps not reached, go back to 'research'.
    - Otherwise proceed to 'review'.
    """
    queue = state.scratch.get("tool_queue", [])
    if queue:
        return "tool"
    if state.steps < MAX_STEPS and state.last_tool_result is not None:
        return "research"
    return "review"


# --------------------------------------------------------------------------------------
# Graph build & run
# --------------------------------------------------------------------------------------
def build_graph(kv: SessionKVStore, checkpointer=None):
    """
    Build and compile the LangGraph state machine, optionally with a checkpointer.
    """
    g = StateGraph(GraphState)

    g.add_node("research", node_research)
    g.add_node("tool", node_tool)
    g.add_node("review", node_review)
    g.add_node("guard", node_guard)

    g.add_edge(START, "research")
    g.add_conditional_edges("research", router, {"tool": "tool", "review": "review"})
    g.add_conditional_edges("tool", router, {"tool": "tool", "research": "research", "review": "review"})
    g.add_edge("review", "guard")
    g.add_edge("guard", END)

    return g.compile(checkpointer=checkpointer)


def run_graph(app_graph, session_id: str, user_message: str, kv: SessionKVStore) -> Dict[str, Any]:
    """
    Invoke the compiled graph for a single user turn and persist session scratch.

    Returns a plain dict with keys:
      session_id, output, guardrail_flags, steps, last_tool_result, history
    """
    # Restore per-session scratch memory
    prior = kv.read(session_id)

    # Seed the state with a Message object for the user input
    initial = GraphState(
        session_id=session_id,
        history=[Message(role="user", content=user_message)],
        scratch=prior,
    )

    # Invoke the graph; depending on LangGraph version, this may return a pydantic
    # model or a plain dict. Normalize to dict immediately.
    raw = app_graph.invoke(initial, config={"configurable": {"thread_id": session_id}})
    state_dict = _result_to_dict(raw)

    # Persist updated scratch (handle both dict and object cases)
    scratch = state_dict.get("scratch", {}) if isinstance(state_dict, dict) else {}
    kv.write(session_id, scratch if isinstance(scratch, dict) else {})

    # Prepare a compact response payload
    out = {
        "session_id": session_id,
        "output": state_dict.get("output"),
        "guardrail_flags": state_dict.get("guardrail_flags", []) or [],
        "steps": state_dict.get("steps", 0) or 0,
        "last_tool_result": state_dict.get("last_tool_result"),
        "history": _history_to_dicts(state_dict.get("history", []))[-6:],
    }
    return out

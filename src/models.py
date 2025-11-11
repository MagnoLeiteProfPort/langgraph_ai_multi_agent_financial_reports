"""
models.py
---------
Pydantic models used by the API layer and LangGraph state.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class RunRequest(BaseModel):
    session_id: str = Field(..., description="Stable session/thread id for memory & checkpointing")
    question: str = Field(..., description="User's research question or task")

class ToolCall(BaseModel):
    tool_name: str
    args: Dict[str, Any]

class Message(BaseModel):
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    meta: Dict[str, Any] = Field(default_factory=dict)

class GraphState(BaseModel):
    session_id: str
    history: List[Message] = Field(default_factory=list)
    scratch: Dict[str, Any] = Field(default_factory=dict)
    last_tool_result: Optional[Dict[str, Any]] = None
    output: Optional[str] = None
    guardrail_flags: List[str] = Field(default_factory=list)
    steps: int = 0

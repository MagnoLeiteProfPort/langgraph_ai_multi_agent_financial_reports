"""
main.py
-------
FastAPI app exposing a simple /run endpoint that executes the LangGraph pipeline.
Includes /health for liveness checks.
"""
from __future__ import annotations
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from ..models import RunRequest
from .deps import get_checkpointer, get_kv, get_graph
from ..graph.graph import run_graph

app = FastAPI(title="LangGraph Finance Research Agents", version="1.0.0")

class RunResponse(BaseModel):
    """Response model for /run endpoint."""
    session_id: str
    output: str | None
    guardrail_flags: list[str]
    steps: int
    last_tool_result: dict | None

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/run", response_model=RunResponse)
def run(
    req: RunRequest,
    checkpointer = Depends(get_checkpointer),  # kept for health/side effects if needed
    kv = Depends(get_kv),
    graph = Depends(get_graph),
):
    """
    Execute a single run through the LangGraph pipeline.
    The request provides a session_id (for continuity) and a question/task.
    """
    result = run_graph(graph, req.session_id, req.question, kv)
    return RunResponse(
        **{k: result[k] for k in ["session_id", "output", "guardrail_flags", "steps", "last_tool_result"]}
    )

# LangGraph Finance Research Agents (AGNO → LangGraph)

Production-grade multi-agent system using **LangGraph** with:
- Session-aware memory & checkpointing (SQLite)
- ReAct-style tool use
- Self-critique (Reviewer agent)
- Guardrails (prompt-time & output-time filters)
- Strong, configurable prompts
- FastAPI demo for event showfloor

> This repository is a structured reimplementation of the original *AGNO AI* showcase,
> ported to LangGraph with professional comments and layered architecture.

## Quickstart (Windows-friendly)

1) **Python 3.11+** and **git** installed.
2) Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
```

3) Install dependencies:

```powershell
pip install -r requirements.txt
```

4) Create your `.env` from the example and set keys:

```powershell
copy .env.example .env
# Set OPENAI_API_KEY, TAVILY_API_KEY (optional), etc.
```

5) Initialize the database (SQLite checkpointer is created on first run).

6) Run the API demo (FastAPI + Uvicorn):

```powershell
uvicorn src.app.main:app --host 127.0.0.1 --port 8787 --reload --access-log
```

7) Try the demo routes:
- `GET  /health`
- `POST /run` with body:
```json
{
  "session_id": "demo-session-1",
  "question": "Give me a brief equity research summary of NVIDIA with key risks."
}
```

## Project Layout

```
langgraph_agno_finance/
├─ requirements.txt
├─ .env.example
├─ README.md
├─ src/
│  ├─ app/
│  │  ├─ main.py
│  │  └─ deps.py
│  ├─ graph/
│  │  ├─ __init__.py
│  │  ├─ graph.py
│  │  ├─ nodes.py
│  │  ├─ memory.py
│  │  ├─ prompts.py
│  │  ├─ guardrails.py
│  │  └─ tools/
│  │     ├─ finance_tools.py
│  │     ├─ web_tools.py
│  │     └─ critique.py
│  ├─ config.py
│  └─ models.py
├─ scripts/
│  └─ run_api.ps1
└─ tests/
   └─ test_smoke.py
```

## Design Notes

- **LangGraph Orchestration**: The graph orchestrates a *Researcher* agent that can call tools (web/news, quotes/prices), then hands results to a *Reviewer* for self-critique. A *Guardrail* node validates outputs before returning.
- **State & Memory**: `SQLiteSaver` checkpointing keyed by `session_id` to persist thread state. A lightweight long-term store is provided for per-session scratch memory (JSON via SQLite).
- **Prompts**: Centralized and versioned in `prompts.py`.
- **Guardrails**: Prompt-time instruction and output validator to ensure safety and formatting.
- **ReAct**: The Researcher produces rationale + actions (tool calls), tools return observations, and the agent iterates when needed.

> Replace the placeholder tool implementations with your preferred providers (e.g., yfinance, Alpha Vantage, Polygon, or internal data).

## License

MIT (example). Review and adapt for your organization.

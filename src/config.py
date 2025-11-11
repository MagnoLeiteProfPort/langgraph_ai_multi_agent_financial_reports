"""
config.py
-----------
Typed configuration loader for environment variables, pathing, and constants.
This centralizes settings so other modules can import a single authoritative source.
"""
from __future__ import annotations
import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    tavily_api_key: str = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    checkpoint_db: str = Field(default_factory=lambda: os.getenv("CHECKPOINT_DB", "./data/checkpoints.sqlite"))
    persist_dir: str = Field(default_factory=lambda: os.getenv("PERSIST_DIR", "./data/persist"))
    project_name: str = Field(default_factory=lambda: os.getenv("LANGCHAIN_PROJECT", "agno-finance-langgraph"))
    tracing: str = Field(default_factory=lambda: os.getenv("LANGCHAIN_TRACING_V2", ""))

    def ensure_dirs(self) -> None:
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_dirs()

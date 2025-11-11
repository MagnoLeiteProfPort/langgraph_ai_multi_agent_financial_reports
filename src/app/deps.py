from __future__ import annotations
from functools import lru_cache
from ..config import settings
from ..graph.memory import make_checkpointer, SessionKVStore
from ..graph.graph import build_graph

@lru_cache(maxsize=1)
def get_checkpointer():
    return make_checkpointer(settings.checkpoint_db)

@lru_cache(maxsize=1)
def get_kv():
    return SessionKVStore(settings.persist_dir)

@lru_cache(maxsize=1)
def get_graph():
    kv = get_kv()
    cp = get_checkpointer()  # <-- actual saver object (not a ctx manager)
    return build_graph(kv, checkpointer=cp)

"""
tools/web_tools.py
------------------
Web & news search tools. Replace with Tavily, NewsAPI, etc.
"""
from __future__ import annotations
from typing import Dict, Any, List

def search_news(query: str, top_k: int = 3) -> Dict[str, Any]:
    # Placeholder: return synthetic news items
    items: List[Dict[str, str]] = [
        {"title": f"{query} - Analyst upgrades outlook",
         "url": "https://example.com/news1",
         "snippet": "Analysts are more optimistic after strong Q3 results."},
        {"title": f"{query} - Supply chain update",
         "url": "https://example.com/news2",
         "snippet": "Management reports improving lead times and demand visibility."},
        {"title": f"{query} - Regulatory watch",
         "url": "https://example.com/news3",
         "snippet": "No material changes expected near-term, monitoring ongoing."},
    ]
    return {"query": query, "results": items}

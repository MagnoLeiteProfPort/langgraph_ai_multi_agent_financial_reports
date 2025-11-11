"""
tools/finance_tools.py
----------------------
Finance-related tools. These are placeholders; swap with your real providers.
All tools return dicts to make them machine-friendly for the graph.
"""
from __future__ import annotations
from typing import Dict, Any, List
import datetime as dt

def get_price(symbol: str) -> Dict[str, Any]:
    # Placeholder; integrate yfinance/AlphaVantage/Polygon as needed.
    # We return a fake price with timestamp to keep the demo self-contained.
    return {
        "symbol": symbol.upper(),
        "price": 123.45,
        "currency": "USD",
        "as_of": dt.datetime.utcnow().isoformat() + "Z",
        "source": "placeholder"
    }

def key_metrics(symbol: str) -> Dict[str, Any]:
    # Placeholder financial metrics
    return {
        "symbol": symbol.upper(),
        "pe_ratio": 28.7,
        "market_cap_usd_b": 250.1,
        "rev_growth_yoy_pct": 22.5,
        "notes": "Synthetic metrics for demo; connect to real data source."
    }

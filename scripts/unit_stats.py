"""What one enemy unit is worth in elixir, and its hit points.

Read from data/unit_stats.json (scripts/build_unit_stats.py, from the game
data). BuildABot's own card table names the units a card brings but not how
many, so it prices one Skeleton of a 1-elixir trio at a full elixir - and
two Skeletons looked worth a 3-elixir Archers.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

STATS = Path(__file__).resolve().parents[1] / "data" / "unit_stats.json"


@lru_cache(maxsize=1)
def _table() -> dict:
    try:
        return json.loads(STATS.read_text(encoding="utf-8")).get("units", {})
    except (OSError, ValueError):
        return {}


def unit_value(name: str, default: float = 0.0) -> float:
    """Elixir one `name` is worth: its card's cost shared by the units it brings."""
    stats = _table().get(name.replace("-", "_"))
    if not stats:
        return default
    return stats["cost"] / max(1, stats["count"])


def unit_hp(name: str) -> Optional[int]:
    """Hit points of one `name` at the opponents' usual level, None when unknown."""
    stats = _table().get(name.replace("-", "_"))
    return int(stats["hp"]) if stats and stats.get("hp") else None

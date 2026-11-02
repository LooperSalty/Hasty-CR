"""The opponent's elixir, inferred from what they deploy.

Nothing on screen shows it. Both players start a match on 5, cap at 10 and
regenerate one every 2.8 s - every 1.4 s from 2:00 and 0.9 s from 3:00. The
opponent is charged a card's cost each time it is seen being played:

* a new enemy *track* that first appears in their half (unit_tracker decides
  that, so one of our troops misread as theirs is never billed), and
* an enemy spell the detector saw that we did not cast ourselves.

A card that deploys several units (3 Minions, 2 Archers) shows up as several
tracks within a fraction of a second: the same unit name is charged once per
DEDUPE_SECONDS. Prices come from BuildABot's card table, which covers every
card the bot can recognise.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

START_ELIXIR = 5.0
MAX_ELIXIR = 10.0
REGEN_SECONDS = {1.0: 2.8, 2.0: 1.4, 3.0: 0.9}
DOUBLE_ELIXIR_SECOND = 120
TRIPLE_ELIXIR_SECOND = 240     # overtime's last minute, as sim/match.py
DEDUPE_SECONDS = 1.5


@lru_cache(maxsize=1)
def _cost_table() -> dict[str, int]:
    """Unit or spell name -> elixir cost of the card that brings it."""
    from clashroyalebuildabot.namespaces.cards import Cards
    table: dict[str, int] = {}
    for name in dir(Cards):
        card = getattr(Cards, name)
        cost = getattr(card, "cost", None)
        if not isinstance(cost, (int, float)) or name.startswith("_"):
            continue
        names = [card.name] + [unit.name for unit in getattr(card, "units", ())]
        for key in names:
            # The cheapest card that brings it: a lone Skeleton is not an Army.
            table[key] = min(int(cost), table.get(key, 99))
    return table


def card_cost(name: str) -> int:
    """Elixir cost for a unit or spell name, 0 when unknown."""
    key = name.replace("-", "_")
    return _cost_table().get(key, 0)


class EnemyElixir:
    def __init__(self):
        self.started_at: Optional[float] = None
        self.elixir = START_ELIXIR
        self.updated_at = 0.0
        self.last_charge: dict[str, float] = {}
        self.recent: list[str] = []

    def start(self, now: float) -> None:
        """A new match: back to 5."""
        self.started_at, self.updated_at = now, now
        self.elixir = START_ELIXIR
        self.last_charge.clear()
        self.recent = []

    def multiplier(self, now: float) -> float:
        elapsed = 0.0 if self.started_at is None else now - self.started_at
        if elapsed >= TRIPLE_ELIXIR_SECOND:
            return 3.0
        if elapsed >= DOUBLE_ELIXIR_SECOND:
            return 2.0
        return 1.0

    def value_at(self, now: float) -> float:
        """Regenerate up to `now` and return the estimate."""
        if self.started_at is None:
            self.start(now)
        elapsed = max(0.0, now - self.updated_at)
        self.updated_at = now
        regen = REGEN_SECONDS[self.multiplier(now)]
        self.elixir = min(MAX_ELIXIR, self.elixir + elapsed / regen)
        return round(self.elixir, 2)

    def charge(self, name: str, now: float) -> int:
        """Bill the opponent for `name` being played; returns what was charged."""
        if now - self.last_charge.get(name, -1e9) < DEDUPE_SECONDS:
            self.last_charge[name] = now          # the rest of the same card's units
            return 0
        cost = card_cost(name)
        self.last_charge[name] = now
        if cost <= 0:
            return 0
        self.value_at(now)
        self.elixir = max(0.0, self.elixir - cost)
        self.recent = ([f"{name} (-{cost})"] + self.recent)[:6]
        return cost

    @property
    def value(self) -> float:
        return round(self.elixir, 2)

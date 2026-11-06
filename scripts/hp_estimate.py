"""How much life an enemy in our half has left, guessed from what hits it.

BuildABot reads no health bars, so the bot sent a card at a troop our tower
had all but killed (user, 2026-11-05: "des placements inutiles car la troupe
n'a plus de vie"). The guess: a unit starts at its hit points from the game
data (unit_stats) and loses, every frame,

* TOWER_DPS from each of our princess towers still standing, which shoots
  the nearest enemy in its range, and
* TROOP_DPS for each of our troops close enough to hit it (one that cannot
  hit air does not count against a flier).

It is rough - no splash, no level, no shields - but a Knight our tower has
shot for ten seconds is not the threat it was when it crossed the river.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from unit_stats import unit_hp

# Princess towers in BuildABot tiles, and their fire at the account's level
# (king 7: 88 damage every 0.8 s, 7.5 tiles - sim towers at level 7).
OUR_TOWERS = {"left": (3.5, 6.5), "right": (14.5, 6.5)}
TOWER_DPS = 110.0
TOWER_RANGE = 7.5
TROOP_DPS = 120.0           # about a Knight's at our levels; the Mini P.E.K.K.A hits harder
TROOP_REACH = 2.0           # tiles: one of ours this close is hitting it
MAX_FRAME_SECONDS = 0.5     # a stalled frame does not count as seconds of fire
NEARLY_DEAD_SHARE = 0.3     # below this share of its life, the towers finish it


def _distance(track, point) -> float:
    return math.hypot(track.x - point[0], track.y - point[1])


class HpEstimator:
    def __init__(self, flies: Callable[[str], bool] = lambda name: False,
                 hits_air: Callable[[str], bool] = lambda name: True):
        self.flies, self.hits_air = flies, hits_air
        self.damage: dict[int, float] = {}
        self.last: Optional[float] = None

    def reset(self) -> None:
        self.damage, self.last = {}, None

    def update(self, enemies: list, allies: list, towers_alive: dict, now: float) -> None:
        dt = 0.0 if self.last is None else min(MAX_FRAME_SECONDS, max(0.0, now - self.last))
        self.last = now
        if dt == 0.0:
            return
        for lane, spot in OUR_TOWERS.items():
            if not towers_alive.get(lane, True):
                continue
            in_range = [t for t in enemies if _distance(t, spot) <= TOWER_RANGE]
            if in_range:
                target = min(in_range, key=lambda t: _distance(t, spot))
                self.damage[target.id] = self.damage.get(target.id, 0.0) + TOWER_DPS * dt
        for enemy in enemies:
            air = self.flies(enemy.name)
            hitting = sum(1 for ally in allies if (not air or self.hits_air(ally.name))
                          and math.hypot(ally.x - enemy.x, ally.y - enemy.y) <= TROOP_REACH)
            if hitting:
                self.damage[enemy.id] = self.damage.get(enemy.id, 0.0) + hitting * TROOP_DPS * dt

    def left(self, track) -> float:
        """Share of its life `track` still has, 1.0 when its hit points are unknown."""
        hp = unit_hp(track.name)
        if not hp:
            return 1.0
        return max(0.0, 1.0 - self.damage.get(track.id, 0.0) / hp)

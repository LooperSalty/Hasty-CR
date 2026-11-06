"""Which enemy in our half is worth a card, which card, and where.

The beatdown plan wants elixir for the Giant, so defence answers only what
the towers cannot handle alone, with the cheapest card that does the job:

* small fry (a Skeleton, a Goblin, a spirit, a Bat) is left to the towers,
  unless SWARM_SIZE of them come together;
* a unit that goes for buildings (Hog Rider, Giant, Ram, Balloon...) is the
  first to stop, before anything nearer;
* the answer is sized to the threat: Mini P.E.K.K.A for a tank or a tower
  rusher, the Knight for a lone fighter or a ground swarm, something that
  hits air for air;
* a threat already fighting one of our troops gets no second card.

Placement: melee in front of the threat in its path, ranged behind it toward
the King and out of its path, Minions straight onto it. Our half is tile
rows 0-14, so every tile is clamped there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

SMALL_UNITS = frozenset({
    "skeleton", "bat", "goblin", "spear_goblin", "ice_spirit", "fire_spirit",
    "electro_spirit", "heal_spirit", "lava_pup", "elixir_golem_small", "phoenix_egg",
    "phoenix_small", "rascal_girl"})
AIR_TANKS = frozenset({
    "balloon", "lava_hound", "baby_dragon", "inferno_dragon", "electro_dragon",
    "skeleton_dragon", "phoenix", "phoenix_large", "mega_minion"})
# Walk to the towers and soak damage: answered first, with the Mini P.E.K.K.A.
# (BuildABot lists the Giant as targeting ground, so `target` alone misses it.)
HEAVY_UNITS = frozenset({
    "giant", "golem", "pekka", "mega_knight", "giant_skeleton", "prince", "dark_prince",
    "royal_giant", "electro_giant", "goblin_giant", "elixir_golem", "ice_golem",
    "elixir_golem_large", "royal_hog", "ram_rider"})
# Go for the towers whatever stands near them (BuildABot often reads them as
# targeting ground).
TOWER_RUSHERS = frozenset({"hog_rider", "battle_ram", "balloon", "wall_breaker"})
SWARM_SIZE = 3              # this many small units together are worth a card
SWARM_RADIUS = 3.0          # tiles: small units this close count as one group
ENGAGE_RADIUS = 4.0         # one of ours this close is already fighting it
NOT_DEFENDERS = frozenset({"giant"})   # walks past troops to the towers
# Cheapest card that does the job first.
DEFENDERS = {
    # The Inferno Tower hits air too, and melts a Balloon or a Lava Hound.
    "air_tank": ("inferno_tower", "musketeer", "archers", "minions"),
    "air": ("archers", "minions", "musketeer"),
    # The Inferno Tower melts a tank pulled to the middle (the user's deck).
    "rusher": ("inferno_tower", "minipekka", "knight", "musketeer", "archers", "minions"),
    "swarm": ("knight", "minions", "archers", "musketeer"),
    "fighter": ("knight", "archers", "minipekka", "musketeer", "minions"),
}
MELEE = frozenset({"knight", "minipekka"})
OVERHEAD = frozenset({"minions"})
# Buildings go in the middle, a few rows from the river, so a tank heading for
# either princess tower turns to them first.
BUILDINGS = frozenset({"inferno_tower", "cannon", "tesla", "bomb_tower"})
BUILDING_ROW = 10        # user 2026-11-01: 8 was too low, the tank reached the towers first
# Ranged defenders stay out of a melee threat's reach (user: a Musketeer was
# dropped on a Mini P.E.K.K.A and died at once).
MELEE_THREATS = frozenset({"knight", "minipekka", "pekka", "valkyrie", "barbarian",
    "elite_barbarian", "angry_barbarian", "mega_knight", "prince", "dark_prince",
    "bandit", "assassin", "lumberjack", "axe_man", "goblin", "skeleton", "guard",
    "royal_recruit", "golden_knight", "skeleton_king", "giant_skeleton", "fisherman"})
MELEE_RANGED_STANDOFF = 6
KNIGHT_COLUMNS = (8, 9)  # user: the Knight goes in the middle and pulls the threat there
MELEE_STANDOFF = 2          # rows in front of the threat, in its path
RANGED_STANDOFF = 4         # rows behind it, toward the King...
RANGED_SIDESTEP = 2         # ...and a little toward the middle, out of its path
LAST_OWN_ROW = 14


@dataclass(frozen=True)
class Threat:
    det: object             # the enemy detection to answer
    kind: str               # a key of DEFENDERS

    @property
    def name(self) -> str:
        return self.det.unit.name


def _attr(det, field: str) -> str:
    return str(getattr(det.unit, field, "") or "").lower()


def flies(det) -> bool:
    return _attr(det, "transport") == "air"


def _targets_buildings(det) -> bool:
    return (_attr(det, "target") == "buildings" or det.unit.name in HEAVY_UNITS
            or det.unit.name in TOWER_RUSHERS)


def _hits_air(det) -> bool:
    return _attr(det, "target") in ("all", "air")


def _distance(a, b) -> float:
    return math.hypot(a.position.tile_x - b.position.tile_x,
                      a.position.tile_y - b.position.tile_y)


def _closest(dets):
    return min(dets, key=lambda det: det.position.tile_y)


def assess(threats: list) -> Optional[Threat]:
    """The enemy in our half most worth a card, or None if the towers will do."""
    big = [det for det in threats if det.unit.name not in SMALL_UNITS]
    if big:
        rushers = [det for det in big if _targets_buildings(det)]
        det = _closest(rushers or big)
        if flies(det):
            return Threat(det, "air_tank" if det.unit.name in AIR_TANKS else "air")
        return Threat(det, "rusher" if rushers else "fighter")
    for det in threats:
        group = [other for other in threats if _distance(det, other) <= SWARM_RADIUS]
        if len(group) >= SWARM_SIZE:
            return Threat(_closest(group), "air" if all(flies(d) for d in group) else "swarm")
    return None


def engaged(threat: Threat, allies: list) -> bool:
    """One of our troops that can hit it is already on it.

    Never for what goes for buildings: a Hog or a Balloon runs past troops to
    the tower, so a Knight standing near it is not stopping it.
    """
    if threat.kind in ("rusher", "air_tank"):
        return False
    air = flies(threat.det)
    return any(ally.unit.name not in NOT_DEFENDERS and (not air or _hits_air(ally))
               and _distance(ally, threat.det) <= ENGAGE_RADIUS
               for ally in allies)


def pick_defender(threat: Threat, ready: dict[str, int],
                  counters: Optional[dict] = None) -> Optional[str]:
    """The first ready card of the threat's own counter list, then of its kind's."""
    order = tuple((counters or {}).get(threat.name, ())) + DEFENDERS[threat.kind]
    return next((name for name in order if name in ready), None)


def _clamp(value: float, low: int, high: int) -> int:
    return int(min(high, max(low, round(value))))


def defence_tile(card: str, threat) -> tuple[int, int]:
    """Where `card` answers `threat` (a detection in our half)."""
    x, y = threat.position.tile_x, threat.position.tile_y
    if card in BUILDINGS:
        return (8 if x <= 8 else 9), BUILDING_ROW
    if card in OVERHEAD:
        return _clamp(x, 0, 17), _clamp(y, 1, LAST_OWN_ROW)
    # The Knight pulls a fighter to the middle; a tower rusher (Hog...) ignores
    # troops, so against one it still goes in front of it, in its lane.
    if card == "knight" and not _targets_buildings(threat):
        return KNIGHT_COLUMNS[0 if x <= 8 else 1], _clamp(y - MELEE_STANDOFF, 1, LAST_OWN_ROW)
    if card in MELEE:
        return _clamp(x, 0, 17), _clamp(y - MELEE_STANDOFF, 1, LAST_OWN_ROW)
    toward_middle = RANGED_SIDESTEP if x <= 8 else -RANGED_SIDESTEP
    back = MELEE_RANGED_STANDOFF if threat.unit.name in MELEE_THREATS else RANGED_STANDOFF
    return _clamp(x + toward_middle, 1, 16), _clamp(y - back, 1, LAST_OWN_ROW)

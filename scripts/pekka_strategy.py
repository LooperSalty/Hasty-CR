"""The user's game plan for the P.E.K.K.A deck (2026-11-01), as rules.

* P.E.K.K.A behind the King Tower at full elixir, on the side of the weaker
  enemy tower: the big push, and the priority.
* Behind it, Musketeer, then Minions, then Archers (the Mini P.E.K.K.A last).
* The P.E.K.K.A never goes in on defence, even with nothing else to answer.
* The Knight defends (beatdown_defence) and is never sent to attack or cycled.
* No troops at the bridge outside a push: between pushes elixir is banked.
* The Inferno Tower answers tanks from the middle (beatdown_defence).
* Arrows only on a group of small troops, wherever it is.

The beatdown engine does the rest: elixir banking, support behind the walking
tank, one defender per wave and a budget per threat.
"""

from __future__ import annotations

import math
from typing import Optional

from beatdown_defence import HEAVY_UNITS, SMALL_UNITS, TOWER_RUSHERS, assess
from beatdown_strategy import BeatdownStrategy, Plan, _clamp, _ready_cards

PEKKA = "pekka"
SUPPORT = ("musketeer", "minions", "archers", "minipekka")
CYCLE = ("archers", "minions")                   # never the Knight
# What Arrows are for. Minions and Archers die to them too.
ARROW_TARGETS = SMALL_UNITS | {"minion", "archer", "goblin", "spear_goblin", "skeleton",
                               "bat", "wall_breaker", "princess", "dart_goblin"}
ARROW_MIN_UNITS = 3
# A tank or a tower rusher is answered while it walks to the bridge, not once
# it is already in our half (user, 2026-11-01 and again 2026-11-05: the
# Inferno Tower came too late - a Hog crosses the river in a second or two).
# Until it crosses only a building is placed (BeatdownStrategy._defence_plan).
TANK_ALERT_ROW = 22
EARLY_THREATS = HEAVY_UNITS | TOWER_RUSHERS | {"royal_hog", "ram_rider", "royal_giant"}
ARROW_RADIUS = 2.5          # tiles: Arrows cover about this far from their centre
# A tank "seen" on their side this close to one of our troops is most likely
# that troop misread (our Mini P.E.K.K.A read as a P.E.K.K.A).
MISREAD_TILES = 2.0


def _distance(a, b) -> float:
    return math.hypot(a.position.tile_x - b.position.tile_x, a.position.tile_y - b.position.tile_y)


class PekkaStrategy(BeatdownStrategy):
    name = "pekka"
    handles_spells = True            # Arrows only on small troops: the runner's rule hits anything
    tank_priority = True             # the big P.E.K.K.A push first, even with the policy attacking
    alert_row = TANK_ALERT_ROW
    early_threats = EARLY_THREATS
    threat_order = True              # enemies in our half before tanks on their way

    def __init__(self):
        super().__init__(tank=PEKKA, tank_label="P.E.K.K.A", support=SUPPORT, cycle=CYCLE,
                         tank_defends=False)

    def _threats(self, state) -> list:
        """Enemies in our half first; a tank on its way only when none of them needs a card.

        Live 2026-11-05: a Mega Knight still at their bridge held our elixir
        while a Bomber shot our tower, and a "pekka" read beside our own
        pushing Mini P.E.K.K.A held it for nothing.
        """
        threats = super()._threats(state)
        if self.threat_order and assess(threats) is not None:
            return threats
        early = [det for det in state.enemies if det.unit.name in self.early_threats
                 and det.position.tile_y <= self.alert_row and det not in threats
                 and not (self.threat_order and any(_distance(det, ally) <= MISREAD_TILES
                                                    for ally in state.allies))]
        return threats + early

    def _spell_plan(self, state) -> Optional[Plan]:
        ready = _ready_cards(state)
        if "arrows" not in ready:
            return None
        small = [det for det in state.enemies if det.unit.name in ARROW_TARGETS]
        best = None
        for centre in small:
            group = [det for det in small if _distance(det, centre) <= ARROW_RADIUS]
            if len(group) >= ARROW_MIN_UNITS and (best is None or len(group) > len(best)):
                best = group
        if best is None:
            return None
        x = sum(det.position.tile_x for det in best) / len(best)
        y = sum(det.position.tile_y for det in best) / len(best)
        tile = (_clamp(x, 0, 17), _clamp(y, 0, 31))
        names = ", ".join(sorted({det.unit.name for det in best}))
        return Plan(phase="sort", card="arrows", slot=ready["arrows"], tile=tile,
                    reason=f"Flèches sur un groupe de petites troupes ({names}) en {tile}")


# Who answers what, before the threat kind's generic list. Live after the
# 2026-11-04 fixes the rules lost 7 of 8 matches with an enemy Wizard: the
# "fighter" list sends Archers or Minions next, which splash damage wipes.
# Splash on the ground: something in the air, or one sturdy melee unit.
# Splash in the air: the Musketeer or the Inferno Tower, never the Minions.
# Melee that cannot hit air: the Minions first.
PEKKA_COUNTERS = {
    "wizard": ("minipekka", "knight", "musketeer"),
    "witch": ("minipekka", "knight", "musketeer"),
    "executioner": ("minipekka", "knight", "musketeer"),
    "bowler": ("minions", "minipekka", "knight"),
    "valkyrie": ("minions", "minipekka", "musketeer", "knight"),
    "bomber": ("minions", "knight", "musketeer", "archers"),
    "mega_knight": ("inferno_tower", "minions", "minipekka"),
    "dark_prince": ("minions", "minipekka", "inferno_tower", "knight"),
    "barbarian": ("minions", "knight", "musketeer"),
    "elite_barbarian": ("minions", "minipekka", "knight"),
    "lumberjack": ("minions", "minipekka", "knight"),
    "axe_man": ("minions", "minipekka", "knight"),
    "golden_knight": ("minions", "minipekka", "knight"),
    "baby_dragon": ("musketeer", "inferno_tower", "archers"),
    "skeleton_dragon": ("musketeer", "inferno_tower", "archers"),
    "electro_dragon": ("musketeer", "inferno_tower", "archers"),
    "inferno_dragon": ("musketeer", "archers", "minions"),
    "balloon": ("inferno_tower", "musketeer", "minions", "archers"),
    "lava_hound": ("inferno_tower", "musketeer", "minions", "archers"),
}


class PekkaCountersStrategy(PekkaStrategy):
    """The P.E.K.K.A plan with a counter per threat (runner `--strategy pekka2`)."""

    name = "pekka2"
    counters = PEKKA_COUNTERS


class PekkaClassicStrategy(PekkaCountersStrategy):
    """pekka2 without the 2026-11-05 defence additions (runner `--strategy pekka2c`).

    The day those went in the rules won 40% (17-25) against 54% (13-11) the
    day before: this plays them switched off - no life estimate, no price
    check, tanks alerted from row 19 only, no elixir held for the Inferno
    Tower, every enemy frame shown - so a live A/B can tell.
    """

    name = "pekka2c"
    skip_nearly_dead = False
    price_check = False
    hold_for_building = False
    alert_row = 19
    early_threats = HEAVY_UNITS
    threat_order = False
    enemy_min_hits = 1               # read by the runner: perception shows one-frame enemies


class PekkaYesterdayStrategy(PekkaStrategy):
    """The P.E.K.K.A rules as they won 13-11 on 2026-11-04 (runner `--strategy pekka_hier`).

    Every 2026-11-05 change switched off - no counters, no life estimate, no
    price check, no early hold, global gap, ties pushed left, buildings read
    in our half defended against, and the tracker not told our buildings are
    ours - to tell a code regression (this wins again) from the evening's
    opponents (it loses too).
    """

    name = "pekka_hier"
    skip_nearly_dead = False
    price_check = False
    hold_for_building = False
    gap_per_lane = False
    lane_by_troops = False
    ignore_structures = False
    alert_row = 19
    early_threats = HEAVY_UNITS
    threat_order = False
    enemy_min_hits = 1
    own_structures = False

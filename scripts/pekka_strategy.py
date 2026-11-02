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

from beatdown_defence import HEAVY_UNITS, SMALL_UNITS
from beatdown_strategy import BeatdownStrategy, Plan, _clamp, _ready_cards

PEKKA = "pekka"
SUPPORT = ("musketeer", "minions", "archers", "minipekka")
CYCLE = ("archers", "minions")                   # never the Knight
# What Arrows are for. Minions and Archers die to them too.
ARROW_TARGETS = SMALL_UNITS | {"minion", "archer", "goblin", "spear_goblin", "skeleton",
                               "bat", "wall_breaker", "princess", "dart_goblin"}
ARROW_MIN_UNITS = 3
# A tank is answered as it reaches the bridge, not once it is already in
# our half (user: the Inferno Tower came too late).
TANK_ALERT_ROW = 19
ARROW_RADIUS = 2.5          # tiles: Arrows cover about this far from their centre


def _distance(a, b) -> float:
    return math.hypot(a.position.tile_x - b.position.tile_x, a.position.tile_y - b.position.tile_y)


class PekkaStrategy(BeatdownStrategy):
    name = "pekka"
    handles_spells = True            # Arrows only on small troops: the runner's rule hits anything

    def __init__(self):
        super().__init__(tank=PEKKA, tank_label="P.E.K.K.A", support=SUPPORT, cycle=CYCLE,
                         tank_defends=False)

    def _threats(self, state) -> list:
        threats = super()._threats(state)
        early = [det for det in state.enemies if det.unit.name in HEAVY_UNITS
                 and det.position.tile_y <= TANK_ALERT_ROW and det not in threats]
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

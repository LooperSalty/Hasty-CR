"""'Gros push' for the starter deck: a Giant from the back, troops behind it.

The plan, as asked for: bank elixir to 10, drop the Giant behind the King
Tower on the side of the weaker enemy tower, then add support troops behind it
as it walks so they arrive at the tower together. Defence is not this
module's main job but it has to happen: when enemies cross into our half,
beatdown_defence picks the one worth a card (small fry is left to the
towers), the cheapest card that answers it and where - never the Giant.
(BuildABot's own rules, used here before, dropped the Knight mid-field and,
at 10 elixir, Archers or Minions wherever, which sent troops up the middle
with no Giant.) The runner fires spells on clusters whatever the phase. Every plan carries a sentence saying why, which the live
view shows as the bot's reasoning.

Tiles are BuildABot's grid: x 0-17 left to right, y 0 at our back row to 14
at the river; enemy-side tiles read y >= 17.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from beatdown_defence import assess, defence_tile, engaged, pick_defender
from enemy_elixir import card_cost

GIANT = "giant"
MAX_ELIXIR = 10
BACK_TILES = {"left": (8, 0), "right": (9, 0)}        # just behind the King Tower
# Behind each princess tower: a cycled card waits there, out of the middle.
CORNER_TILES = {"left": (3, 1), "right": (14, 1)}
LANE_X = {"left": 3, "right": 14}
# A Giant dropped behind the King walks diagonally into its lane, reaching the
# lane's column about this many rows up.
TURN_ROWS = 8.0
# Elixir sitting at 10 this long is being wasted: cycle one cheap card to
# bring the Giant round, rather than sending troops with no tank.
CYCLE_AFTER_FULL_SECONDS = 2.5
LANE_NAMES = {"left": "gauche", "right": "droite"}
# Ranged first: they hit what attacks the Giant from behind it.
SUPPORT_PRIORITY = ("musketeer", "archers", "minions", "minipekka", "knight")
CYCLE_PRIORITY = ("knight", "archers", "minions", "minipekka", "musketeer")
SUPPORT_GAP_TILES = 2           # how far behind the Giant support is dropped
MIN_GIANT_Y_FOR_SUPPORT = 3     # until then there is no room behind it
LAST_OWN_ROW = 14
OUR_SIDE_MAX_Y = 15             # an enemy at or below this row is in our half
GIANT_TILES_PER_SECOND = 0.75   # estimate when vision has not seen it yet
GIANT_LOST_SECONDS = 3          # seen, then not seen for this long: it died
PUSH_TIMEOUT_SECONDS = 35
DEFENCE_GAP_SECONDS = 3.0   # one defender per wave, not the whole hand
# Elixir spent on one threat: defenders in its lane in the last few seconds.
# More than its own cost (plus a little) is a losing trade - live, 16 elixir
# went on a single 5-elixir Giant, a card every 3 s while it walked.
COMMIT_WINDOW_SECONDS = 8.0
COMMIT_LANE_TILES = 5
COMMIT_MARGIN = 1
DEFAULT_THREAT_COST = 4
DEFENCE = "défense"


@dataclass(frozen=True)
class Plan:
    phase: str
    reason: str
    card: Optional[str] = None
    slot: Optional[int] = None
    tile: Optional[tuple[int, int]] = None
    delegate: bool = False                      # let BuildABot's rules defend
    exclude: frozenset = frozenset()            # cards the delegate may not use
    then: Optional["Plan"] = None               # if the delegate finds nothing


@dataclass(frozen=True)
class Push:
    lane: str
    started: float
    giant: tuple[float, float]
    seen_at: Optional[float] = None
    start_x: float = 8.0


def _clamp(value: float, low: int, high: int) -> int:
    return int(min(high, max(low, round(value))))


def _ready_cards(state) -> dict[str, int]:
    """Card name -> hand slot, for cards both ready and affordable now."""
    elixir = state.numbers.elixir.number
    out = {}
    for slot in state.ready:
        card = state.cards[slot + 1]
        if card.name not in out and elixir >= card.cost:
            out[card.name] = slot
    return out


def _weaker_lane(state) -> str:
    numbers = state.numbers
    left, right = numbers.left_enemy_princess_hp.number, numbers.right_enemy_princess_hp.number
    return "right" if right < left else "left"


def _first(priority, ready: dict[str, int]) -> Optional[str]:
    return next((name for name in priority if name in ready), None)


class BeatdownStrategy:
    def __init__(self, tank: str = GIANT, tank_label: str = "Géant",
                 support: tuple = SUPPORT_PRIORITY, cycle: tuple = CYCLE_PRIORITY,
                 tank_defends: bool = True):
        self.tank, self.tank_label = tank, tank_label
        self.support, self.cycle = support, cycle
        # The Giant also tanks for the towers when nothing else answers; a
        # tank kept for pushes (the user's P.E.K.K.A) never goes in on defence.
        self.tank_defends = tank_defends
        self.push: Optional[Push] = None
        self.full_since: Optional[float] = None
        self.last_defence: Optional[float] = None
        self.last_plan: Optional[Plan] = None
        self.defence_log: list[tuple[float, int, int]] = []     # (time, elixir, column)

    def played(self, card: str, tile: tuple[int, int], now: float) -> None:
        """Called once a card was really deployed: a Giant starts a push."""
        if card == self.tank:
            lane = "left" if tile[0] <= 8 else "right"
            self.push = Push(lane=lane, started=now, giant=(float(tile[0]), float(tile[1])),
                             start_x=float(tile[0]))
        planned = self.last_plan
        if planned is not None and planned.phase == DEFENCE and planned.card == card:
            self.last_defence = now
            self.defence_log = ([entry for entry in self.defence_log
                                 if now - entry[0] <= COMMIT_WINDOW_SECONDS]
                                + [(now, card_cost(card), int(tile[0]))])
        self.full_since = None

    def decide(self, state, now: float) -> Plan:
        elixir = state.numbers.elixir.number
        if elixir >= MAX_ELIXIR:
            self.full_since = now if self.full_since is None else self.full_since
        else:
            self.full_since = None
        self._track_giant(state, now)
        plan = self._push_plan(state, now) if self.push else self._build_plan(state, now)
        threats = self._threats(state)
        spell = self._spell_plan(state)
        if spell is not None:
            plan = spell
        elif threats:
            plan = self._defence_plan(state, threats, now, plan)
        self.last_plan = plan
        return plan

    def threats(self, state) -> list:
        """Public for `hybrid_strategy`: the enemies this strategy defends against."""
        return self._threats(state)

    def spell_plan(self, state) -> Optional[Plan]:
        """Public for `hybrid_strategy`: the spell this strategy would cast now."""
        return self._spell_plan(state)

    def _threats(self, state) -> list:
        """Enemies this strategy defends against: those in our half."""
        return [det for det in state.enemies if det.position.tile_y <= OUR_SIDE_MAX_Y]

    def _spell_plan(self, state) -> Optional[Plan]:
        """A spell this strategy casts itself; none here (the runner's rule does)."""
        return None

    # --------------------------------------------------------------- defence

    def defence_plan(self, state, threats: list, now: float, fallback: Plan) -> Plan:
        """The defence for `threats` in our half, or `fallback` when none is due.

        Public for `hybrid_strategy`, where `fallback` is the policy's plan.
        """
        return self._defence_plan(state, threats, now, fallback)

    def _defence_plan(self, state, threats: list, now: float, fallback: Plan) -> Plan:
        threat = assess(threats)
        if threat is None:
            names = ", ".join(sorted({det.unit.name for det in threats}))
            return replace(fallback, reason=f"{names} dans mon camp : les tours s'en chargent"
                                            f" · {fallback.reason}")
        det = threat.det
        where = f"Défense : {threat.name} dans mon camp ({det.position.tile_x:.0f},"\
                f"{det.position.tile_y:.0f})"
        # Covered already: carry on with the plan (the Giant at full elixir,
        # support behind it) rather than sit on a full bar.
        if self.last_defence is not None and now - self.last_defence < DEFENCE_GAP_SECONDS:
            return replace(fallback, reason=f"{where}, défenseur déjà posé · {fallback.reason}")
        if engaged(threat, state.allies):
            return replace(fallback, reason=f"{where}, déjà engagé par mes troupes · "
                                            f"{fallback.reason}")
        committed = self._committed(det, now)
        budget = (card_cost(threat.name) or DEFAULT_THREAT_COST) + COMMIT_MARGIN
        if committed >= budget:
            return replace(fallback, reason=f"{where}, {committed} élixir déjà engagé dessus · "
                                            f"{fallback.reason}")
        ready = _ready_cards(state)
        ready.pop(self.tank, None)
        card = pick_defender(threat, ready)
        if card is None:
            elixir = state.numbers.elixir.number
            if fallback.card == self.tank and self.tank_defends:
                # Nothing to answer with but a full bar: the Giant still goes,
                # behind the King, where it also tanks for the towers.
                return replace(fallback, reason=f"{where}, aucun défenseur prêt : {fallback.reason}")
            return Plan(phase=DEFENCE,
                        reason=f"{where}, aucun défenseur prêt ou pas assez d'élixir "
                               f"({elixir}/10) : les tours défendent")
        tile = defence_tile(card, det)
        return Plan(phase=DEFENCE, card=card, slot=ready[card], tile=tile,
                    reason=f"{where}, je pose {card} en {tile}")

    def _committed(self, threat, now: float) -> int:
        x = threat.position.tile_x
        return sum(cost for when, cost, column in self.defence_log
                   if now - when <= COMMIT_WINDOW_SECONDS and abs(column - x) <= COMMIT_LANE_TILES)

    # ------------------------------------------------------------ push state

    def _track_giant(self, state, now: float) -> None:
        push = self.push
        if push is None:
            return
        giants = [det for det in state.allies if det.unit.name == self.tank]
        if giants:
            gx, gy = push.giant
            nearest = min(giants, key=lambda det: abs(det.position.tile_x - gx)
                          + abs(det.position.tile_y - gy))
            self.push = replace(push, giant=(nearest.position.tile_x, nearest.position.tile_y),
                                seen_at=now)
            return
        lost = push.seen_at is not None and now - push.seen_at > GIANT_LOST_SECONDS
        if lost or now - push.started > PUSH_TIMEOUT_SECONDS:
            self.push = None
        elif push.seen_at is None:
            # Not seen yet: walk the estimate from behind the King diagonally
            # into its lane, as the Giant itself does - so support follows it
            # into the lane instead of filing up the middle.
            walked = min(15.0, (now - push.started) * GIANT_TILES_PER_SECOND)
            lane_x = LANE_X[push.lane]
            x = push.start_x + (lane_x - push.start_x) * min(1.0, walked / TURN_ROWS)
            self.push = replace(push, giant=(x, walked))

    # ----------------------------------------------------------------- plans

    def _push_plan(self, state, now: float) -> Plan:
        push = self.push
        elixir = state.numbers.elixir.number
        gx, gy = push.giant
        lane = LANE_NAMES[push.lane]
        where = f"Push voie {lane} : {self.tank_label} en y≈{gy:.0f}"
        if gy < MIN_GIANT_Y_FOR_SUPPORT:
            return Plan(phase="push", reason=f"{where}, je le laisse avancer avant le soutien")
        ready = _ready_cards(state)
        support = _first(self.support, ready)
        if support is None:
            return Plan(phase="push",
                        reason=f"{where}, j'attends l'élixir pour le soutenir ({elixir}/10)")
        tile = (_clamp(gx, 0, 17), _clamp(gy - SUPPORT_GAP_TILES, 1, LAST_OWN_ROW))
        return Plan(phase="push", card=support, slot=ready[support], tile=tile,
                    reason=f"{where}, je pose {support} derrière le {self.tank_label} {tile}")

    def _build_plan(self, state, now: float) -> Plan:
        elixir = state.numbers.elixir.number
        ready = _ready_cards(state)
        lane = _weaker_lane(state)
        if self.tank in ready and elixir >= MAX_ELIXIR:
            return Plan(phase="lancement", card=self.tank, slot=ready[self.tank], tile=BACK_TILES[lane],
                        reason=f"Élixir plein : {self.tank_label} au fond, voie {LANE_NAMES[lane]} "
                               f"(tour adverse la plus faible)")
        in_hand = any(card.name == self.tank for card in state.cards[1:5])
        wasted = self.full_since is not None and now - self.full_since >= CYCLE_AFTER_FULL_SECONDS
        if wasted and not in_hand:
            # One cheap card, parked behind a princess tower: it brings the
            # Giant one step closer without sending a push with no tank.
            cycle = _first(self.cycle, ready)
            if cycle is not None:
                return Plan(phase="cycle", card=cycle, slot=ready[cycle], tile=CORNER_TILES[lane],
                            reason=f"Élixir plein et pas de {self.tank_label} en main : je fais tourner "
                                   f"{cycle} dans le coin, pour récupérer le {self.tank_label}")
        waiting = (f"le {self.tank_label} est en main" if in_hand
                   else f"j'attends que le {self.tank_label} revienne en main")
        return Plan(phase="économie",
                    reason=f"J'économise pour le {self.tank_label} ({elixir}/10), {waiting}")

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

from beatdown_defence import BUILDINGS, assess, defence_tile, engaged, pick_defender
from enemy_elixir import card_cost
from hp_estimate import NEARLY_DEAD_SHARE
from unit_stats import unit_value
from unit_tracker import STRUCTURES

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
LANE_HP_GAP = 0.05              # enemy towers closer than this in life count as equal
MIDDLE_COLUMNS = (7, 10)        # a defender dropped here covers both lanes
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
# Enemies within this many tiles (|dx| + |dy|) of a threat count as its group.
GROUP_TILES = 4
# What goes for our towers is stopped whatever it costs; anything else is not
# answered with more elixir than it is worth (user: Archers for 2 Skeletons).
ALWAYS_ANSWERED = frozenset({"rusher", "air_tank"})
# A defender may cost this much more than the group: a Knight on three
# Goblins (1.5) survives them and walks on; Archers on two Skeletons do not pay.
VALUE_MARGIN = 1.5
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


def _first(priority, ready: dict[str, int]) -> Optional[str]:
    return next((name for name in priority if name in ready), None)


def _same_lane(column: float, x: float, middle_covers: bool = True) -> bool:
    """A defender dropped at `column` covers a threat at `x`: same lane, or the
    middle, where the Knight and the Inferno Tower go for either lane (live
    2026-11-05: a Knight at column 8 did not count for a threat at x=1, and a
    second defender followed at once)."""
    middle = MIDDLE_COLUMNS[0] <= column <= MIDDLE_COLUMNS[1]
    return abs(column - x) <= COMMIT_LANE_TILES or (middle_covers and middle)


def _nearly_dead(det) -> bool:
    return getattr(det, "hp_left", 1.0) < NEARLY_DEAD_SHARE


def _group_value(det, threats: list) -> float:
    """Elixir the enemies around `det` are worth (two Skeletons: two thirds)."""
    x, y = det.position.tile_x, det.position.tile_y
    group = [other for other in threats
             if abs(other.position.tile_x - x) + abs(other.position.tile_y - y) <= GROUP_TILES]
    return sum(unit_value(other.unit.name, default=card_cost(other.unit.name)
                          or DEFAULT_THREAT_COST) for other in group)


class BeatdownStrategy:
    # With a trained policy attacking (hybrid_strategy), the tank at full
    # elixir still goes first: the user's P.E.K.K.A plan, not the Giant's.
    tank_priority = False
    # The 2026-11-05 defence rules, each switchable for a live A/B against
    # the rules as they were the day before (pekka_strategy "pekka2c").
    skip_nearly_dead = True         # no card on a troop our fire has all but killed
    price_check = True              # no defender dearer than the group it answers
    hold_for_building = True        # a threat on their side: only a building, elixir kept
    gap_per_lane = True             # the gap after a defender holds in its lane only
    lane_by_troops = True           # equal towers: push where our troops are, or switch lanes
    ignore_structures = True        # a building read in our half is never a threat
    # Threat name -> defenders in order, before the threat kind's list.
    counters: Optional[dict] = None

    def __init__(self, tank: str = GIANT, tank_label: str = "Géant",
                 support: tuple = SUPPORT_PRIORITY, cycle: tuple = CYCLE_PRIORITY,
                 tank_defends: bool = True):
        self.tank, self.tank_label = tank, tank_label
        self.support, self.cycle = support, cycle
        # The Giant also tanks for the towers when nothing else answers; a
        # tank kept for pushes (the user's P.E.K.K.A) never goes in on defence.
        self.tank_defends = tank_defends
        self.push: Optional[Push] = None
        self.last_push_lane: Optional[str] = None
        self.full_since: Optional[float] = None
        self.last_defence: Optional[float] = None
        self.last_plan: Optional[Plan] = None
        self.defence_log: list[tuple[float, int, int]] = []     # (time, elixir, column)

    def played(self, card: str, tile: tuple[int, int], now: float) -> None:
        """Called once a card was really deployed: a Giant starts a push."""
        if card == self.tank:
            lane = "left" if tile[0] <= 8 else "right"
            self.last_push_lane = lane
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
        """Enemies this strategy defends against: those in our half.

        Never a building: the opponent cannot build there, so one read there
        is a misread (ours, or scenery) and troops sent at it are wasted.
        Never one our towers and troops have all but killed (hp_estimate).
        """
        return [det for det in state.enemies if det.position.tile_y <= OUR_SIDE_MAX_Y
                and not (self.ignore_structures and det.unit.name in STRUCTURES)
                and not (self.skip_nearly_dead and _nearly_dead(det))]

    def _spell_plan(self, state) -> Optional[Plan]:
        """A spell this strategy casts itself; none here (the runner's rule does)."""
        return None

    # --------------------------------------------------------------- defence

    def defence_plan(self, state, threats: list, now: float, fallback: Plan) -> Plan:
        """The defence for `threats` in our half, or `fallback` when none is due.

        Public for `hybrid_strategy`, where `fallback` is the policy's plan.
        """
        return self._defence_plan(state, threats, now, fallback)

    def open_threat(self, state, threats: list, now: float):
        """The threat still waiting for a card, or None (small fry, or covered).

        Public for `hybrid_strategy`: a tank kept for pushes may go once
        nothing is left unanswered.
        """
        threat = assess(threats)
        # A defender just dropped is not fighting yet: it covers the next
        # card, not the tank, until our troops are on the threat.
        if threat is None or self._covered(state, threat, now, gap=False) is not None:
            return None
        return threat

    def launch_plan(self, state) -> Optional[Plan]:
        """The tank behind the King at full elixir, if this rule set puts it first."""
        return self._launch(state) if self.tank_priority else None

    def _covered(self, state, threat, now: float, gap: bool = True) -> Optional[str]:
        """Why `threat` needs no new card now, or None if it does.

        The gap after a defender holds in that defender's lane only: a tank
        coming down the other lane is answered at once.
        """
        x = threat.det.position.tile_x
        # A defender in the middle pulls a fighter from either lane, but does
        # not stop a rusher running down the other one.
        middle_covers = threat.kind not in ALWAYS_ANSWERED
        if gap and any(now - when < DEFENCE_GAP_SECONDS
                       and (not self.gap_per_lane or _same_lane(column, x, middle_covers))
                       for when, _cost, column in self.defence_log):
            return "défenseur déjà posé"
        if engaged(threat, state.allies):
            return "déjà engagé par mes troupes"
        committed = self._committed(threat.det, now)
        budget = (card_cost(threat.name) or DEFAULT_THREAT_COST) + COMMIT_MARGIN
        if committed >= budget:
            return f"{committed} élixir déjà engagé dessus"
        return None

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
        covered = self._covered(state, threat, now)
        if covered is not None:
            return replace(fallback, reason=f"{where}, {covered} · {fallback.reason}")
        ready = _ready_cards(state)
        ready.pop(self.tank, None)
        card = pick_defender(threat, ready, self.counters)
        if card is None:
            elixir = state.numbers.elixir.number
            if fallback.card == self.tank and self.tank_defends:
                # Nothing to answer with but a full bar: the Giant still goes,
                # behind the King, where it also tanks for the towers.
                return replace(fallback, reason=f"{where}, aucun défenseur prêt : {fallback.reason}")
            return Plan(phase=DEFENCE,
                        reason=f"{where}, aucun défenseur prêt ou pas assez d'élixir "
                               f"({elixir}/10) : les tours défendent")
        if self.hold_for_building and det.position.tile_y > OUR_SIDE_MAX_Y and card not in BUILDINGS:
            # Still on their side (a tank or a rusher seen early): only a
            # building goes now; troops would walk out to meet it at their
            # tower. Elixir is kept for the Inferno Tower if it is in hand -
            # with none to come, the plan goes on (live: a full bar sat idle).
            in_hand = {c.name for c in state.cards[1:5]}
            if in_hand & BUILDINGS:
                return Plan(phase=DEFENCE, reason=f"{where} : il arrive, je garde l'élixir pour "
                                                  f"la Tour de l'enfer")
            if fallback.card == self.tank and not self.tank_defends:
                return Plan(phase=DEFENCE, reason=f"{where} : il arrive, pas de "
                                                  f"{self.tank_label} sur sa route")
            return replace(fallback, reason=f"{where} : il arrive, j'attends qu'il traverse · "
                                            f"{fallback.reason}")
        value = _group_value(det, threats)
        if (self.price_check and threat.kind not in ALWAYS_ANSWERED
                and card_cost(card) > value + VALUE_MARGIN):
            return replace(fallback, reason=f"{where}, {value:.1f} élixir de troupes : {card} "
                                            f"({card_cost(card)}) coûte trop, les tours s'en "
                                            f"chargent · {fallback.reason}")
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

    def _push_lane(self, state) -> tuple[str, str]:
        """(lane, why): the weaker enemy tower; with both alike, where our
        troops already are (a counter-push), else not the lane of the last
        push (live 2026-11-05: 17 pushes out of 17 went left - both towers
        read 1.0 at the start and ties went left)."""
        numbers = state.numbers
        left = numbers.left_enemy_princess_hp.number
        right = numbers.right_enemy_princess_hp.number
        if abs(left - right) >= LANE_HP_GAP or not self.lane_by_troops:
            return ("right" if right < left else "left"), "tour adverse la plus faible"
        ours = {"left": 0, "right": 0}
        for ally in state.allies:
            ours["left" if ally.position.tile_x <= 8 else "right"] += 1
        if ours["left"] != ours["right"]:
            return max(ours, key=ours.get), "avec mes troupes déjà sur place"
        if self.last_push_lane is not None:
            return ("right" if self.last_push_lane == "left" else "left"), "pas la même voie qu'avant"
        return "left", "tours adverses égales"

    def _launch(self, state) -> Optional[Plan]:
        ready = _ready_cards(state)
        if self.tank not in ready or state.numbers.elixir.number < MAX_ELIXIR:
            return None
        lane, why = self._push_lane(state)
        return Plan(phase="lancement", card=self.tank, slot=ready[self.tank], tile=BACK_TILES[lane],
                    reason=f"Élixir plein : {self.tank_label} au fond, voie {LANE_NAMES[lane]} ({why})")

    def _build_plan(self, state, now: float) -> Plan:
        elixir = state.numbers.elixir.number
        ready = _ready_cards(state)
        lane = self._push_lane(state)[0]
        launch = self._launch(state)
        if launch is not None:
            return launch
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

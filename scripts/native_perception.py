"""The native runner's view of the board, one frame at a time.

detect -> track -> side -> enemy elixir:

* units come from YOLO (or BuildABot's own model if YOLO is off or failing),
* unit_tracker follows them across frames and fixes each one's side where it
  first appeared, so a troop of ours the side classifier flips for a frame is
  not suddenly a threat,
* the opponent is charged for each new troop that appears on their side of
  the field and for every spell the detector sees that we did not cast -
  but not for units another one brought: a Witch's skeletons, a Golem's
  golemites, the goblins out of a Goblin Barrel. Priced as the cheapest card
  that brings them (Golem 8, Lava Hound 7), those drained the estimate to 0.

The result is written back into BuildABot's State (allies / enemies), so the
strategies and actions downstream are unchanged.
"""

from __future__ import annotations

import math
from typing import Optional

from enemy_elixir import EnemyElixir
from hp_estimate import HpEstimator
from unit_tracker import DEPLOY_ANYWHERE, Detection, UnitTracker

TOWER_TROOPS = frozenset({"dagger_duchess_tower"})
# An enemy is shown to the strategies once its track has lasted this many
# frames: a one-frame misread is not a threat (live 2026-11-05: a "giant"
# flashed where our Archers had just landed and took the Inferno Tower).
ENEMY_MIN_HITS = 3
# ...and this many when it was first seen deep in our half, where no enemy
# lands: a real one walked in from the bridge on a track of its own.
DEEP_BORN_MIN_HITS = 8
DEEP_BORN_Y = 12.0
# A troop first seen deeper in our half than this was re-acquired after a
# missed detection, not deployed - deploys land in their half or at the bridge.
ENEMY_SPAWN_MIN_Y = 12.0
OWN_SPELL_SECONDS = 3.0          # a spell box this soon after we cast it is ours
# Unit -> the enemies that spawn it (alive, dying or dead). A new one near one
# of those was brought by it, not deployed; near anything else it was played.
# "elixir_golem" covers both the whole golem and its halves (one class).
SPAWNERS = {
    "skeleton": frozenset({"witch", "tombstone", "giant_skeleton", "skeleton_king"}),
    "bat": frozenset({"night_witch"}),
    "golemite": frozenset({"golem"}),
    "lava_pup": frozenset({"lava_hound"}),
    "elixir_golem": frozenset({"elixir_golem", "elixir_golem_large"}),
    "elixir_golem_medium": frozenset({"elixir_golem", "elixir_golem_large"}),
    "elixir_golem_small": frozenset({"elixir_golem", "elixir_golem_medium"}),
    "goblin": frozenset({"goblin_drill"}),
    "spear_goblin": frozenset({"goblin_hut", "goblin_giant"}),
    "barbarian": frozenset({"barbarian_hut"}),
    "phoenix_egg": frozenset({"phoenix", "phoenix_large"}),
    "phoenix_small": frozenset({"phoenix_egg"}),
}
SPAWNER_RADIUS = 4.0
# Spells that land units: those units were paid for with the spell.
SPELL_SPAWNS = {"goblin_barrel": "goblin", "skeleton_barrel": "skeleton",
                "graveyard": "skeleton", "barbarian_barrel": "barbarian"}
SPELL_SPAWN_SECONDS = 4.0
# The arena in BuildABot tiles. BuildABot's model reads the side decorations
# as a Knight at column -2 or 19 now and then: outside these, nothing is real.
ARENA_X = (0.0, 17.0)
ARENA_Y = (0.0, 31.0)
ARENA_MARGIN = 0.5
# Our King Tower's back row: BuildABot reads its scenery as a unit we do not
# own (a Lava Pup at (9,0) all match, live); no enemy troop stands there.
BACK_ROW_MAX_Y = 0.5
UPSTREAM_BOX_SCALE = 1080 / 368  # BuildABot's boxes live in its 368x652 frame
VISION_FAILURES_BEFORE_FALLBACK = 20


def _in_arena(d) -> bool:
    return (ARENA_X[0] - ARENA_MARGIN <= d.x <= ARENA_X[1] + ARENA_MARGIN
            and ARENA_Y[0] - ARENA_MARGIN <= d.y <= ARENA_Y[1] + ARENA_MARGIN)


def _unit_table() -> dict:
    """Unit name -> BuildABot Unit, the same lookup YoloVision.apply uses."""
    from brain.vision import _extra_units
    from clashroyalebuildabot.namespaces.units import NAME2UNIT
    table = {unit.name: unit for unit in NAME2UNIT.values()}
    table.update(_extra_units())
    return table


class Perception:
    def __init__(self, deck, vision, detector, upstream_units, log,
                 enemy_min_hits: int = ENEMY_MIN_HITS, deep_born_min_hits: int = DEEP_BORN_MIN_HITS,
                 own_structures: bool = True):
        self.enemy_min_hits, self.deep_born_min_hits = enemy_min_hits, deep_born_min_hits
        self.own_structures = own_structures
        self.vision, self.detector, self.upstream_units, self.log = (
            vision, detector, upstream_units, log)
        self.tracker = UnitTracker({unit.name for card in deck for unit in card.units},
                                   own_structures=self.own_structures)
        self.enemy = EnemyElixir()
        self.hp = HpEstimator(flies=self._flies, hits_air=self._hits_air)
        self.own_spells: dict[str, float] = {}
        self.enemy_spells: dict[str, float] = {}   # spawned unit -> when its spell landed
        self.vision_failures = 0
        self._units: Optional[dict] = None

    def start_match(self, now: float) -> None:
        self.tracker.reset()
        self.hp.reset()
        self.enemy.start(now)
        self.own_spells.clear()
        self.enemy_spells.clear()

    def played(self, card: str, now: float, tile: Optional[tuple] = None) -> None:
        self.tracker.played(card, now, tile)
        self.own_spells[card] = now

    # ------------------------------------------------------------------ frame

    def perceive(self, state, full, now: float) -> None:
        detections = self._detections(state, full)
        tracks = self.tracker.update(detections, now)
        self.hp.update([t for t in tracks if t.side == "enemy"],
                       [t for t in tracks if t.side == "ally"], _towers_alive(state), now)
        allies, enemies = [], []
        for track in tracks:
            if track.name in TOWER_TROOPS or track.side == "shadow":
                continue
            ours = track.side == "ally"
            if not ours and track.hits < self._min_hits(track):
                continue
            detection = self._unit_detection(track, state,
                                             hp_left=1.0 if ours else self.hp.left(track))
            if detection is not None:
                (allies if ours else enemies).append(detection)
        state.allies, state.enemies = allies, enemies
        for name, *_ in (getattr(self.vision, "last_spells", None) or []):
            if now - self.own_spells.get(name, -1e9) > OWN_SPELL_SECONDS:
                self.enemy.charge(name, now)
                if name in SPELL_SPAWNS:
                    self.enemy_spells[SPELL_SPAWNS[name]] = now
        for track in self.tracker.spawned:
            if (track.first_y >= ENEMY_SPAWN_MIN_Y and track.name not in TOWER_TROOPS
                    and not self._spawned_by_another(track, tracks, now)):
                self.enemy.charge(track.name, now)
        self.enemy.value_at(now)

    def _spawned_by_another(self, track, tracks, now: float) -> bool:
        if now - self.enemy_spells.get(track.name, -1e9) <= SPELL_SPAWN_SECONDS:
            return True
        spawners = SPAWNERS.get(track.name)
        if not spawners:
            return False
        # Recently lost tracks count too: a Golem's golemites appear as it dies.
        return any(other.side == "enemy" and other.name in spawners and other is not track
                   and other.first_seen < now
                   and math.hypot(other.x - track.x, other.y - track.y) <= SPAWNER_RADIUS
                   for other in list(tracks) + list(self.tracker.lost))

    def _detections(self, state, full) -> list[Detection]:
        return [d for d in self._raw_detections(state, full) if _in_arena(d)
                and not (d.y <= BACK_ROW_MAX_Y and d.name not in self.tracker.our_units)]

    def _raw_detections(self, state, full) -> list[Detection]:
        if self.vision is not None:
            self.vision.error = ""
            ours, theirs = self.vision.detect(full)
            if not self.vision.error:
                self.vision_failures = 0
                return ([self._from_seen(s, "ally") for s in ours]
                        + [self._from_seen(s, "enemy") for s in theirs])
            self._vision_failed()
        # BuildABot's own model: its two lists are the side hint.
        self._remember_units(state)
        return ([self._from_upstream(d, "ally") for d in state.allies]
                + [self._from_upstream(d, "enemy") for d in state.enemies])

    def _vision_failed(self) -> None:
        self.vision_failures += 1
        if self.vision_failures == 1:
            self.log(f"VISION error {self.vision.error}")
        if self.vision_failures >= VISION_FAILURES_BEFORE_FALLBACK and self.upstream_units:
            self.detector.unit_detector = self.upstream_units
            self.vision = None
            self.log(f"VISION fallback to buildabot after {self.vision_failures} failed frames")

    @staticmethod
    def _from_seen(seen, hint: str) -> Detection:
        return Detection(name=seen.name, x=float(seen.tile_x), y=float(seen.tile_y),
                         box=tuple(seen.box) or None, conf=float(seen.confidence),
                         side_hint=hint, side_conf=float(seen.side_confidence))

    @staticmethod
    def _from_upstream(det, hint: str) -> Detection:
        bbox = tuple(det.position.bbox)
        box = tuple(round(v * UPSTREAM_BOX_SCALE) for v in bbox) if any(bbox) else None
        return Detection(name=det.unit.name, x=float(det.position.tile_x),
                         y=float(det.position.tile_y), box=box, conf=float(det.position.conf),
                         side_hint=hint, side_conf=0.6)

    def _remember_units(self, state) -> None:
        table = self._table()
        for det in list(state.allies) + list(state.enemies):
            table.setdefault(det.unit.name, det.unit)

    def _table(self) -> dict:
        if self._units is None:
            try:
                self._units = _unit_table()
            except Exception:
                self._units = {}
        return self._units

    def _unit_detection(self, track, state, hp_left: float = 1.0):
        from clashroyalebuildabot.namespaces.units import Position
        unit = self._table().get(track.name)
        if unit is None:
            return None
        return _tracked_detection()(unit=unit, position=Position(
            bbox=track.box or (0, 0, 0, 0), conf=track.conf,
            tile_x=track.x, tile_y=track.y), hp_left=hp_left)

    def _min_hits(self, track) -> int:
        """Frames an enemy track needs before the strategies see it."""
        deep = track.first_y < DEEP_BORN_Y and track.name not in DEPLOY_ANYWHERE
        return self.deep_born_min_hits if deep else self.enemy_min_hits

    def _attr(self, name: str, field: str) -> str:
        unit = self._table().get(name)
        return str(getattr(unit, field, "") or "").lower()

    def _flies(self, name: str) -> bool:
        return self._attr(name, "transport") == "air"

    def _hits_air(self, name: str) -> bool:
        return self._attr(name, "target") in ("all", "air")


def _towers_alive(state) -> dict:
    """Our princess towers still standing (a missing reading counts as standing)."""
    numbers = getattr(state, "numbers", None)
    alive = {}
    for lane in ("left", "right"):
        reading = getattr(numbers, f"{lane}_ally_princess_hp", None)
        alive[lane] = reading is None or getattr(reading, "number", 1) != 0
    return alive


_TRACKED = None


def _tracked_detection():
    """BuildABot's UnitDetection plus `hp_left`, the share of life we guess it has."""
    global _TRACKED
    if _TRACKED is None:
        from dataclasses import dataclass
        from clashroyalebuildabot.namespaces.units import UnitDetection

        @dataclass(frozen=True)
        class TrackedDetection(UnitDetection):
            hp_left: float = 1.0

        _TRACKED = TrackedDetection
    return _TRACKED

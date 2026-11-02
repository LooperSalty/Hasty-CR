"""Units followed across frames, their side fixed by where they first appeared.

The side classifier judges each frame on its own and is right ~94% of the
time - so in a 10-frames-a-second loop a troop of ours flips to "enemy" every
second or so, which reads as a threat and pulls a defender into the middle.
Clash Royale itself gives a sturdier answer: a player can only deploy in their
own half. So a unit's side is decided once, when its track starts:

* a card we do not own is the opponent's;
* one of ours that appears in our half right after we played that card is
  ours, and one that appears in their half is the opponent's copy;
* anything else falls back to the classifier's verdict.

Tracks also outlive a missed detection by HOLD_SECONDS, so a box does not
vanish for a frame. A track lost for longer is remembered LOST_MEMORY_SECONDS:
a unit of that name turning up within walking distance is the same unit come
back (hidden in a melee, behind a tower), keeping its side and not counted as
a new deploy. Brand-new enemy tracks are reported for the elixir count.

BuildABot often reads one unit as several (our P.E.K.K.A as pekka, minipekka
and giant at once). A name we do not own that first appears deep in our half,
beside one of our units and away from any enemy, is such a misread - the
opponent cannot deploy there - and its track is a "shadow": neither side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import count
from typing import Optional

HOLD_SECONDS = 0.45          # keep a track this long after its last detection
MAX_STEP_TILES = 2.5         # how far a unit may move between two detections
STEP_TILES_PER_SECOND = 4.0  # plus this per second of gap (fast troops)
OUR_HALF_MAX_Y = 14.5        # BuildABot tiles: our half 0-14, river 15-16
THEIR_HALF_MIN_Y = 16.5
OWN_SPAWN_SECONDS = 10.0     # a card we played may still be spawning its units
OWN_SPAWN_RADIUS = 4.0       # ...within this many tiles of where it was dropped,
OWN_WALK_TILES_PER_SECOND = 1.2  # plus how far it has walked since: a Giant
# dropped behind the King Tower is hidden by it and first read ~5 s later,
# tiles away - and then the side classifier called it an enemy.
LOST_MEMORY_SECONDS = 8.0    # a lost track can be picked up again this long
LOST_TILES_PER_SECOND = 2.0  # as fast as the fastest troop really walks
# A unit claimed ours far from where we dropped the card (it was hidden, then
# re-read) that then walks toward our King is the opponent's copy of the same
# card: ours walk toward their towers. Live, an enemy Giant claimed ours this
# way was never defended and took the tower.
FAR_CLAIM_DOWN_TILES = 1.5
# Our unit shows up where we dropped it within a couple of seconds; a unit of
# that name appearing there later can just as well be theirs walking past (the
# AI drops its Giant at the bridge, right where an enemy Giant comes down).
NEAR_CLAIM_SECONDS = 3.0
# These walk to buildings and nothing else: one walking toward our King is never
# ours, whoever said it was - a play claim or the side classifier (live, an
# enemy Giant read 'ally' by BuildABot walked to our tower undefended).
TOWER_WALKERS = frozenset({"giant", "golem", "royal_giant", "goblin_giant", "electro_giant",
                           "elixir_golem", "hog_rider", "ram_rider", "battle_ram",
                           "balloon", "wall_breaker", "lava_hound"})
# Card name -> the unit names it deploys, where they differ.
# A new track this deep in our half beside our unit, with no enemy near, is a
# misread of ours. Kept below the pocket the opponent can drop into once a
# Princess Tower falls.
SHADOW_MAX_Y = 11.0
SHADOW_RADIUS = 2.5
SHADOW_ENEMY_LOST_SECONDS = 2.0  # a just-lost enemy nearby may be what is seen
SHADOW_DROP_SECONDS = 6.0        # ...or near where we dropped a troop this recently
# Units the opponent can put anywhere (spells, burrowers): never a shadow.
DEPLOY_ANYWHERE = frozenset({"miner", "goblin", "skeleton", "royal_recruit",
                             "goblin_drill", "goblin_barrel", "skeleton_barrel",
                             "graveyard", "bat", "royal_delivery"})
CARD_UNITS = {"archers": ("archer",), "minions": ("minion",),
              "skeletons": ("skeleton",), "spear_goblins": ("spear_goblin",),
              "goblins": ("goblin",), "barbarians": ("barbarian",)}


@dataclass(frozen=True)
class Detection:
    name: str
    x: float
    y: float
    box: Optional[tuple]
    conf: float
    side_hint: str                 # "ally" / "enemy", the classifier's verdict
    side_conf: float


@dataclass
class Track:
    id: int
    name: str
    x: float
    y: float
    box: Optional[tuple]
    conf: float
    side: str
    first_y: float
    first_seen: float
    last_seen: float
    hits: int = 1
    missing: bool = False
    far_claim: bool = False        # ours only because we played that card a while ago
    claimed: bool = False          # ours because we played that card (not the classifier)


@dataclass
class UnitTracker:
    our_units: set
    tracks: list = field(default_factory=list)
    spawned: list = field(default_factory=list)
    # unit name -> (time we played it, tile it was dropped on or None)
    own_plays: dict = field(default_factory=dict)
    lost: list = field(default_factory=list)            # dropped tracks, recently
    _ids: count = field(default_factory=lambda: count(1))

    def reset(self) -> None:
        self.tracks, self.spawned, self.own_plays, self.lost = [], [], {}, []

    def played(self, card: str, now: float, tile: Optional[tuple] = None) -> None:
        """We deployed `card` on `tile`: its units appearing there are ours."""
        for unit in CARD_UNITS.get(card, (card,)):
            self.own_plays[unit] = (now, tile)

    def _ours_by_play(self, d: Detection, now: float) -> Optional[str]:
        """'near' (where it was dropped), 'far' (where it could have walked) or None."""
        when, tile = self.own_plays.get(d.name, (-1e9, None))
        if d.y > OUR_HALF_MAX_Y or now - when > OWN_SPAWN_SECONDS:
            return None
        age = max(0.0, now - when)
        if tile is None:
            return "near" if age <= NEAR_CLAIM_SECONDS else "far"
        distance = math.hypot(tile[0] - d.x, tile[1] - d.y)
        if distance <= OWN_SPAWN_RADIUS and age <= NEAR_CLAIM_SECONDS:
            return "near"
        reach = OWN_SPAWN_RADIUS + OWN_WALK_TILES_PER_SECOND * age
        return "far" if distance <= reach else None

    def _side(self, d: Detection, now: float) -> str:
        if d.name not in self.our_units:
            return "enemy"
        if d.y >= THEIR_HALF_MIN_Y:
            return "enemy"
        if self._ours_by_play(d, now):
            return "ally"
        return d.side_hint

    def _dropped_here(self, d: Detection, lost: Track, now: float) -> bool:
        """`d` is our unit just dropped rather than the lost enemy `lost` coming back.

        Both are possible where we drop a card into a fight: whichever spot
        it is nearer to decides, the enemy on a tie (a mirror Knight lost in
        the melee half way between).
        """
        if d.name not in self.our_units or d.y >= THEIR_HALF_MIN_Y:
            return False
        if self._ours_by_play(d, now) != "near":
            return False
        tile = self.own_plays[d.name][1]
        if tile is None:
            return True
        return (math.hypot(tile[0] - d.x, tile[1] - d.y)
                < math.hypot(lost.x - d.x, lost.y - d.y))

    @staticmethod
    def _recheck(track: Track) -> None:
        if track.side == "shadow" and track.y <= track.first_y - FAR_CLAIM_DOWN_TILES:
            track.side = "enemy"             # it walks toward our King: theirs
            return
        if (track.side == "ally" and (track.far_claim or track.name in TOWER_WALKERS)
                and track.y <= track.first_y - FAR_CLAIM_DOWN_TILES):
            track.side, track.far_claim, track.claimed = "enemy", False, False

    def _revive(self, d: Detection, now: float, side: Optional[str] = None) -> Optional[Track]:
        """The recently lost track `d` is most likely the return of, if any.

        `side` restricts it to tracks of that side: a unit appearing where we
        just dropped that card is ours, and must not inherit a lost enemy of
        the same name (live, 14% of our troop drops were read as enemies).
        """
        best = None
        for track in self.lost:
            if track.name != d.name or (side is not None and track.side != side):
                continue
            reach = MAX_STEP_TILES + LOST_TILES_PER_SECOND * max(0.0, now - track.last_seen)
            distance = math.hypot(track.x - d.x, track.y - d.y)
            if distance <= reach and (best is None or distance < best[0]):
                best = (distance, track)
        if best is None:
            return None
        track = best[1]
        self.lost.remove(track)
        return track

    def update(self, detections: list[Detection], now: float) -> list[Track]:
        """Match this frame's detections to tracks; returns the live tracks."""
        pairs = []
        for di, d in enumerate(detections):
            for ti, track in enumerate(self.tracks):
                if track.name != d.name:
                    continue
                gap = max(0.0, now - track.last_seen)
                reach = MAX_STEP_TILES + STEP_TILES_PER_SECOND * gap
                distance = math.hypot(track.x - d.x, track.y - d.y)
                if distance <= reach:
                    pairs.append((distance, di, ti))
        used_d, used_t = set(), set()
        for _, di, ti in sorted(pairs):
            if di in used_d or ti in used_t:
                continue
            used_d.add(di)
            used_t.add(ti)
            d, track = detections[di], self.tracks[ti]
            track.x, track.y, track.box, track.conf = d.x, d.y, d.box, d.conf
            track.last_seen, track.hits, track.missing = now, track.hits + 1, False
            self._recheck(track)
        self.spawned, born = [], []
        for di, d in enumerate(detections):
            if di in used_d:
                continue
            track = self._revive(d, now)
            if track is not None and track.side == "enemy" and self._dropped_here(d, track, now):
                self.lost.append(track)            # not that enemy: ours, just dropped
                track = self._revive(d, now, side="ally")
            if track is not None:
                track.x, track.y, track.box, track.conf = d.x, d.y, d.box, d.conf
                track.last_seen, track.hits, track.missing = now, track.hits + 1, False
                self.tracks.append(track)
                continue
            side = self._side(d, now)
            claim = (self._ours_by_play(d, now) if side == "ally" and d.name in self.our_units
                     and d.y < THEIR_HALF_MIN_Y else None)
            far = claim == "far"
            track = Track(id=next(self._ids), name=d.name, x=d.x, y=d.y, box=d.box,
                          conf=d.conf, side=side, first_y=d.y,
                          first_seen=now, last_seen=now, far_claim=far,
                          claimed=claim is not None)
            self.tracks.append(track)
            born.append(track)
            if track.side == "enemy":
                self.spawned.append(track)
        self._mark_shadows(born, now)
        for track in self.tracks:
            if track.last_seen != now:
                track.missing = True
        kept = [t for t in self.tracks if now - t.last_seen <= HOLD_SECONDS]
        dropped = [t for t in self.tracks if now - t.last_seen > HOLD_SECONDS]
        self.lost = [t for t in self.lost + dropped if now - t.last_seen <= LOST_MEMORY_SECONDS]
        self.tracks = kept
        return list(self.tracks)

    def _mark_shadows(self, born: list[Track], now: float) -> None:
        """Enemy tracks born this frame that are misreads of our own units."""
        for track in born:
            if (track.side != "enemy" or track.name in self.our_units
                    or track.name in DEPLOY_ANYWHERE or track.y > SHADOW_MAX_Y):
                continue
            near = [o for o in self.tracks if o is not track
                    and math.hypot(o.x - track.x, o.y - track.y) <= SHADOW_RADIUS]
            lost_enemy = any(o.side == "enemy" and now - o.last_seen <= SHADOW_ENEMY_LOST_SECONDS
                             and math.hypot(o.x - track.x, o.y - track.y) <= SHADOW_RADIUS
                             for o in self.lost)
            ours = any(o.side == "ally" and o.claimed for o in near) or self._near_our_drop(track, now)
            if ours and not any(o.side == "enemy" for o in near) and not lost_enemy:
                track.side = "shadow"
                self.spawned.remove(track)

    def _near_our_drop(self, track: Track, now: float) -> bool:
        """`track` is where a troop we dropped lately could have walked to."""
        for name, (when, tile) in self.own_plays.items():
            age = now - when
            if name not in self.our_units or tile is None or not 0.0 <= age <= SHADOW_DROP_SECONDS:
                continue
            reach = OWN_SPAWN_RADIUS + OWN_WALK_TILES_PER_SECOND * age
            if math.hypot(tile[0] - track.x, tile[1] - track.y) <= reach:
                return True
        return False

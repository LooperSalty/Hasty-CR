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

A unit of one of our cards that only a late play claim or the classifier
made ours or theirs is not settled at birth: every frame's classifier verdict
is a vote, and a clear majority over enough frames moves it (live, half the
"threats" the P.E.K.K.A deck defended against were names of our own deck,
a Knight walking down to defend flipped to enemy). A troop of ours lost in a
melee and read again deep in our half, where no enemy of that name has been
lately, is ours come back, whatever one frame of the classifier says.

BuildABot also gives one box several names in the same frame (live
2026-11-04, IoU 1.0: knight/valkyrie, archer/skeleton, inferno_tower/furnace,
pekka/minipekka): each name started its own track, and the wrong one was an
enemy. One box keeps one name - the one that continues a track, else the most
confident.
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
# Side votes: once a track has this many frames, a share this large of
# classifier verdicts for the other side moves it there.
VOTE_MIN_FRAMES = 6
VOTE_FLIP_SHARE = 0.7
# Deploys land in their half or at the bridge: a unit first read deeper than
# this was there already, lost and found again.
DEEP_Y = 12.0
DUPLICATE_IOU = 0.8          # two names on boxes this alike are one unit read twice
# Buildings stay where they were put, and the opponent cannot build deep in
# our half: one of our deck's there is ours (live 2026-11-04: our Inferno
# Tower read as the enemy's drew two Knights onto it).
STRUCTURES = frozenset({"inferno_tower", "cannon", "tesla", "bomb_tower", "goblin_cage",
                        "tombstone", "furnace", "goblin_hut", "barbarian_hut",
                        "elixir_collector", "mortar", "xbow"})
REACQUIRE_SECONDS = 40.0     # a troop of ours can still be alive this long after its play
ENEMY_NAME_SECONDS = 15.0    # ...unless an enemy of that name was seen this recently
CARD_UNITS = {"archers": ("archer",), "minions": ("minion",),
              "skeletons": ("skeleton",), "spear_goblins": ("spear_goblin",),
              "goblins": ("goblin",), "barbarians": ("barbarian",)}


def _iou(a: Optional[tuple], b: Optional[tuple]) -> float:
    if not a or not b:
        return 0.0
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


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
    ally_votes: int = 0            # frames the classifier called it ours...
    enemy_votes: int = 0           # ...and theirs

    def vote(self, hint: str) -> None:
        if hint == "ally":
            self.ally_votes += 1
        else:
            self.enemy_votes += 1


@dataclass
class UnitTracker:
    our_units: set
    tracks: list = field(default_factory=list)
    spawned: list = field(default_factory=list)
    # unit name -> (time we played it, tile it was dropped on or None)
    own_plays: dict = field(default_factory=dict)
    lost: list = field(default_factory=list)            # dropped tracks, recently
    enemy_seen: dict = field(default_factory=dict)      # unit name -> last time an enemy one was
    _ids: count = field(default_factory=lambda: count(1))
    own_structures: bool = True     # our deck's buildings deep in our half are ours (STRUCTURES)

    def reset(self) -> None:
        self.tracks, self.spawned, self.own_plays, self.lost = [], [], {}, []
        self.enemy_seen = {}

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

    def _claim(self, d: Detection, now: float) -> Optional[str]:
        """Why `d` is ours: 'near' (for sure), 'far' (until the votes say otherwise) or None."""
        if d.name not in self.our_units or d.y >= THEIR_HALF_MIN_Y:
            return None
        if self.own_structures and d.name in STRUCTURES and d.y < DEEP_Y:
            return "near"
        return self._ours_by_play(d, now) or ("far" if self._reacquired_ours(d, now) else None)

    def _side(self, d: Detection, now: float) -> str:
        if d.name not in self.our_units or d.y >= THEIR_HALF_MIN_Y:
            return "enemy"
        return "ally" if self._claim(d, now) else d.side_hint

    def _reacquired_ours(self, d: Detection, now: float) -> bool:
        """`d`, deep in our half, is a troop of ours lost a while and read again."""
        when, _tile = self.own_plays.get(d.name, (-1e9, None))
        return (d.y < DEEP_Y and 0.0 <= now - when <= REACQUIRE_SECONDS
                and now - self.enemy_seen.get(d.name, -1e9) > ENEMY_NAME_SECONDS)

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

    def _recheck(self, track: Track) -> None:
        walked_down = track.y <= track.first_y - FAR_CLAIM_DOWN_TILES
        if track.side == "shadow" and walked_down:
            track.side = "enemy"             # it walks toward our King: theirs
            return
        if track.side == "ally" and track.name in TOWER_WALKERS and walked_down:
            track.side, track.far_claim, track.claimed = "enemy", False, False
            return
        # Our troops walk down to defend, so walking down proves nothing for
        # the others: what the classifier says frame after frame decides.
        near_claim = track.claimed and not track.far_claim
        if (near_claim or track.name not in self.our_units or track.first_y >= THEIR_HALF_MIN_Y
                or track.side not in ("ally", "enemy")):
            return
        frames = track.ally_votes + track.enemy_votes
        if frames < VOTE_MIN_FRAMES:
            return
        if track.side == "ally" and track.enemy_votes >= VOTE_FLIP_SHARE * frames:
            track.side, track.far_claim, track.claimed = "enemy", False, False
        elif track.side == "enemy" and track.ally_votes >= VOTE_FLIP_SHARE * frames:
            track.side = "ally"

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

    def _continues(self, d: Detection, now: float) -> bool:
        """`d` is where a live track of its name could have moved to."""
        return any(track.name == d.name and math.hypot(track.x - d.x, track.y - d.y)
                   <= MAX_STEP_TILES + STEP_TILES_PER_SECOND * max(0.0, now - track.last_seen)
                   for track in self.tracks)

    def _one_name_per_box(self, detections: list[Detection], now: float) -> list[Detection]:
        kept: list[Detection] = []
        for d in sorted(detections, key=lambda d: -d.conf):
            twin = next((i for i, k in enumerate(kept) if k.name != d.name
                         and _iou(k.box, d.box) >= DUPLICATE_IOU), None)
            if twin is None:
                kept.append(d)
            elif self._continues(d, now) and not self._continues(kept[twin], now):
                kept[twin] = d
        return kept

    def update(self, detections: list[Detection], now: float) -> list[Track]:
        """Match this frame's detections to tracks; returns the live tracks."""
        detections = self._one_name_per_box(detections, now)
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
            track.vote(d.side_hint)
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
                track.vote(d.side_hint)
                self.tracks.append(track)
                continue
            side = self._side(d, now)
            claim = self._claim(d, now) if side == "ally" else None
            track = Track(id=next(self._ids), name=d.name, x=d.x, y=d.y, box=d.box,
                          conf=d.conf, side=side, first_y=d.y,
                          first_seen=now, last_seen=now, far_claim=claim == "far",
                          claimed=claim is not None)
            track.vote(d.side_hint)
            self.tracks.append(track)
            born.append(track)
            if track.side == "enemy":
                self.spawned.append(track)
        self._mark_shadows(born, now)
        for track in self.tracks:
            if track.last_seen != now:
                track.missing = True
            elif track.side == "enemy":
                self.enemy_seen[track.name] = now
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

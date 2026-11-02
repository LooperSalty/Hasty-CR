"""Units followed across frames, their side fixed by where they first appeared.

Tiles are BuildABot's: our half is y 0-14, the river 15-16, theirs 17+.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from unit_tracker import Detection, UnitTracker  # noqa: E402

OURS = {"knight", "archer", "minion", "giant", "musketeer", "minipekka"}


def det(name, x, y, hint="enemy", hint_conf=0.6, conf=0.8):
    return Detection(name=name, x=x, y=y, box=(1, 2, 3, 4), conf=conf,
                     side_hint=hint, side_conf=hint_conf)


def tracker():
    return UnitTracker(OURS)


def test_a_unit_keeps_its_identity_as_it_walks():
    t = tracker()
    first = t.update([det("giant", 8, 1, hint="ally")], 0.0)
    later = t.update([det("giant", 8, 2.2, hint="ally")], 0.5)
    assert len(later) == 1 and later[0].id == first[0].id
    assert later[0].y == 2.2


def test_our_troop_stays_ours_even_when_the_classifier_flips():
    t = tracker()
    t.played("musketeer", 0.0)
    t.update([det("musketeer", 4, 5, hint="ally")], 0.1)
    y = 5.0
    for step in range(1, 13):                   # walks across the river into their half
        y += 1.1
        walking = t.update([det("musketeer", 4, y, hint="enemy", hint_conf=0.9)], 0.1 + step * 0.4)
    assert walking[0].y > 17
    assert walking[0].side == "ally"            # side was fixed where it appeared


def test_a_card_we_do_not_own_is_always_the_enemy():
    t = tracker()
    tracks = t.update([det("hog_rider", 14, 9, hint="ally", hint_conf=0.95)], 0.0)
    assert tracks[0].side == "enemy"


def test_our_card_spawning_in_their_half_is_the_opponents_copy():
    t = tracker()
    tracks = t.update([det("knight", 9, 25, hint="ally", hint_conf=0.6)], 0.0)
    assert tracks[0].side == "enemy"


def test_our_card_appearing_in_our_half_right_after_we_played_it_is_ours():
    t = tracker()
    t.played("archers", 3.0)                   # the card name, not the unit name
    tracks = t.update([det("archer", 7, 10, hint="enemy", hint_conf=0.7)], 3.4)
    assert tracks[0].side == "ally"


def test_a_flickering_unit_is_held_briefly_then_dropped():
    t = tracker()
    t.update([det("giant", 9, 25)], 0.0)
    held = t.update([], 0.3)
    assert len(held) == 1 and held[0].missing
    assert t.update([], 0.8) == []


def test_new_enemy_spawns_are_reported_once():
    t = tracker()
    t.update([det("minion", 5, 26), det("minion", 6, 26), det("minion", 7, 26)], 0.0)
    assert sorted(s.name for s in t.spawned) == ["minion", "minion", "minion"]
    t.update([det("minion", 5, 25), det("minion", 6, 25), det("minion", 7, 25)], 0.2)
    assert t.spawned == []


def test_an_enemy_lost_in_a_fight_comes_back_as_the_same_enemy():
    """Mirror match: we drop a Knight, theirs vanishes for a second in the
    melee and reappears in our half. It is their Knight, already paid for."""
    t = tracker()
    t.update([det("knight", 9, 18)], 0.0)
    t.update([det("knight", 9, 14)], 2.0)
    t.played("knight", 2.5, tile=(9, 11))
    t.update([], 3.0)
    t.update([], 3.6)                           # dropped after HOLD_SECONDS
    back = t.update([det("knight", 9, 12.5, hint="ally", hint_conf=0.7)], 4.2)
    assert [track.side for track in back] == ["enemy"]
    assert t.spawned == []                      # not a new deploy


def test_our_play_claims_only_units_near_where_it_landed():
    t = tracker()
    t.played("knight", 0.0, tile=(3, 8))
    far = t.update([det("knight", 14, 12, hint="enemy", hint_conf=0.8)], 0.8)
    assert far[0].side == "enemy"
    near = t.update([det("knight", 14, 12, hint="enemy", hint_conf=0.8),
                     det("knight", 3, 9, hint="enemy", hint_conf=0.8)], 1.0)
    assert sorted(track.side for track in near) == ["ally", "enemy"]


def test_reset_forgets_everything():
    t = tracker()
    t.update([det("giant", 9, 25)], 0.0)
    t.reset()
    assert t.update([], 0.1) == []


def test_our_giant_seen_only_once_it_walks_out_from_behind_the_king_is_ours():
    """Dropped at (8,0), the Giant is hidden by the King Tower and first read
    ~5 s later, several tiles away - and the side classifier then called it
    an enemy: the bot defended against its own Giant."""
    t = tracker()
    t.played("giant", 0.0, tile=(8, 0))
    tracks = t.update([det("giant", 5, 5, hint="enemy", hint_conf=0.8)], 5.5)
    assert tracks[0].side == "ally"


def test_a_late_far_unit_of_our_card_is_not_claimed_forever():
    t = tracker()
    t.played("giant", 0.0, tile=(8, 0))
    tracks = t.update([det("giant", 14, 13, hint="enemy", hint_conf=0.8)], 20.0)
    assert tracks[0].side == "enemy"


def test_an_enemy_giant_reacquired_in_our_half_is_not_ours_for_long():
    """Live 2026-10-12: we played our Giant, theirs was lost then re-read deep
    in our half within reach of our play - claimed ours, never defended, the
    tower fell. A far claim that walks toward our King is the enemy."""
    t = tracker()
    t.played("giant", 0.0, tile=(14, 13))
    t.update([det("giant", 12, 10, hint="enemy", hint_conf=0.6)], 5.0)
    t.update([det("giant", 12, 8.8, hint="enemy", hint_conf=0.6)], 6.0)
    tracks = t.update([det("giant", 11, 7.5, hint="enemy", hint_conf=0.6)], 7.0)
    assert [track.side for track in tracks] == ["enemy"]


def test_our_far_claimed_giant_walking_up_stays_ours():
    t = tracker()
    t.played("giant", 0.0, tile=(8, 0))
    t.update([det("giant", 5, 5, hint="enemy", hint_conf=0.8)], 5.5)
    tracks = t.update([det("giant", 4.5, 7, hint="enemy", hint_conf=0.8)], 7.5)
    assert [track.side for track in tracks] == ["ally"]


def test_an_enemy_lost_for_six_seconds_is_still_recognised():
    t = tracker()
    t.update([det("giant", 14, 18)], 0.0)
    t.update([det("giant", 14, 15)], 3.0)
    t.played("giant", 4.0, tile=(14, 13))
    for now in (4.0, 5.0, 6.0, 7.0, 8.0, 9.0):       # frames where it is not seen
        t.update([], now)
    back = t.update([det("giant", 13, 9, hint="ally", hint_conf=0.6)], 10.0)
    assert [track.side for track in back] == ["enemy"]


def test_a_giant_walking_toward_our_king_is_never_ours():
    """We dropped our Giant at the bridge just as theirs came down that lane:
    both read 'ours' (near the drop, in time). Ours walks up; theirs down."""
    t = tracker()
    t.played("giant", 44.0, tile=(14, 13))
    t.update([det("giant", 13, 12, hint="ally"), det("giant", 12, 10, hint="enemy")], 47.0)
    tracks = t.update([det("giant", 14, 15, hint="ally"), det("giant", 11, 7.5, hint="enemy")], 50.0)
    assert sorted((round(tr.y), tr.side) for tr in tracks) == [(8, "enemy"), (15, "ally")]


def test_our_knight_walking_down_to_defend_stays_ours():
    t = tracker()
    t.played("knight", 0.0, tile=(4, 12))
    t.update([det("knight", 4, 12, hint="ally")], 0.5)
    tracks = t.update([det("knight", 4, 9, hint="enemy")], 3.0)
    assert [tr.side for tr in tracks] == ["ally"]


def test_a_giant_the_classifier_calls_ours_but_walks_to_our_king_is_theirs():
    """Live 21:28: an enemy Giant read 'ally' by BuildABot's side detector,
    minutes after our last Giant - no play claim involved at all."""
    t = tracker()
    t.update([det("giant", 11, 9, hint="ally", hint_conf=0.6)], 100.0)
    t.update([det("giant", 11, 8, hint="ally", hint_conf=0.6)], 101.0)
    tracks = t.update([det("giant", 11, 6.5, hint="ally", hint_conf=0.6)], 102.0)
    assert [tr.side for tr in tracks] == ["enemy"]


def test_our_giant_dropped_behind_the_king_walking_up_stays_ours():
    t = tracker()
    t.played("giant", 0.0, tile=(2, 3))
    t.update([det("giant", 2, 3, hint="ally")], 0.8)
    tracks = t.update([det("giant", 2.5, 7, hint="ally")], 5.0)
    assert [tr.side for tr in tracks] == ["ally"]


def test_our_fresh_troop_is_not_revived_as_a_lost_enemy_of_the_same_name():
    """Live, 14% of our troop drops showed an 'enemy' of that same type where
    we had just dropped it (3% for a type we had not played). A lost enemy
    track of the same name could be revived onto our new unit - up to 8 s
    later and ~15 tiles away - before the play claim was even looked at."""
    t = tracker()
    t.update([det("archer", 4, 12, hint="enemy")], 0.0)        # their archers at our bridge
    t.update([], 1.0)                                          # killed / lost
    t.played("archers", 2.0, (3, 13))
    tracks = t.update([det("archer", 3, 13, hint="enemy")], 3.0)
    assert [tr.side for tr in tracks] == ["ally"]


def test_a_lost_enemy_still_comes_back_where_we_played_nothing():
    t = tracker()
    t.update([det("archer", 4, 12, hint="enemy")], 0.0)
    t.update([], 1.0)
    tracks = t.update([det("archer", 4, 11, hint="ally")], 2.0)
    assert [tr.side for tr in tracks] == ["enemy"]


# --- our own unit read under another name ("shadow") ------------------------
# Live 2026-11-01: right after we dropped the P.E.K.K.A behind our King Tower,
# BuildABot read it at once as pekka, minipekka AND giant; the "giant" (not in
# our deck, so the opponent's) drew Archers and Minions onto our own push.

PEKKA_OURS = {"knight", "pekka", "minion", "archer", "musketeer", "minipekka"}


def test_a_misread_of_our_unit_deep_in_our_half_is_a_shadow_not_an_enemy():
    t = UnitTracker(PEKKA_OURS)
    t.played("pekka", 0.0, (8, 0))
    tracks = t.update([det("pekka", 7, 4, hint="ally"), det("giant", 8, 6)], 3.0)
    sides = {tr.name: tr.side for tr in tracks}
    assert sides == {"pekka": "ally", "giant": "shadow"}
    assert t.spawned == []
    later = t.update([det("pekka", 7, 5, hint="ally"), det("giant", 8, 7)], 3.3)
    assert {tr.name: tr.side for tr in later}["giant"] == "shadow"


def test_a_new_name_beside_a_real_enemy_stays_an_enemy():
    """Next to an enemy it may be that enemy misread: defend rather than ignore."""
    t = UnitTracker(PEKKA_OURS)
    t.played("knight", 0.0, (11, 8))
    t.update([det("knight", 11, 8, hint="ally"), det("valkyrie", 11, 13)], 0.5)
    t.update([det("knight", 11, 8, hint="ally"), det("valkyrie", 11, 11.5)], 1.0)
    tracks = t.update([det("knight", 11, 8, hint="ally"), det("valkyrie", 11, 10),
                       det("giant", 11, 9)], 1.5)
    assert {tr.name: tr.side for tr in tracks}["giant"] == "enemy"


def test_a_shadow_walking_toward_our_king_is_theirs_after_all():
    t = UnitTracker(PEKKA_OURS)
    t.played("knight", 0.0, (8, 7))
    first = t.update([det("knight", 8, 7, hint="ally"), det("giant", 8, 8)], 0.5)
    assert {tr.name: tr.side for tr in first}["giant"] == "shadow"
    tracks = t.update([det("knight", 8, 7, hint="ally"), det("giant", 8, 6.3)], 1.0)
    assert {tr.name: tr.side for tr in tracks}["giant"] == "enemy"


def test_no_shadow_near_the_river_where_the_opponent_can_drop_after_a_tower():
    t = UnitTracker(PEKKA_OURS)
    t.played("knight", 0.0, (3, 12))
    t.update([det("knight", 3, 12, hint="ally")], 0.0)
    tracks = t.update([det("knight", 3, 12, hint="ally"), det("giant", 3, 13)], 0.2)
    assert {tr.name: tr.side for tr in tracks}["giant"] == "enemy"


def test_units_the_opponent_can_drop_anywhere_are_never_shadows():
    t = UnitTracker(PEKKA_OURS)
    t.played("knight", 0.0, (8, 6))
    t.update([det("knight", 8, 6, hint="ally")], 0.0)
    tracks = t.update([det("knight", 8, 6, hint="ally"), det("miner", 8, 7)], 0.2)
    assert {tr.name: tr.side for tr in tracks}["miner"] == "enemy"


def test_a_shadow_lost_for_a_moment_comes_back_a_shadow():
    t = UnitTracker(PEKKA_OURS)
    t.played("pekka", -2.0, (8, 0))
    t.update([det("pekka", 7, 4, hint="ally"), det("giant", 8, 6)], 0.0)
    t.update([det("pekka", 7, 5, hint="ally")], 1.0)
    tracks = t.update([det("pekka", 7, 6, hint="ally"), det("giant", 8, 8)], 2.0)
    assert {tr.name: tr.side for tr in tracks}["giant"] == "shadow"


def test_no_shadow_beside_a_unit_only_the_classifier_calls_ours():
    """That 'ally' may itself be an enemy misread: only units we dropped count."""
    t = UnitTracker(PEKKA_OURS)
    t.update([det("knight", 8, 6, hint="ally")], 0.0)
    tracks = t.update([det("knight", 8, 6, hint="ally"), det("giant", 8, 7)], 0.2)
    assert {tr.name: tr.side for tr in tracks}["giant"] == "enemy"


def test_a_misread_where_our_troop_was_just_dropped_is_a_shadow_on_its_own():
    """Behind the King Tower the P.E.K.K.A is often seen only as 'giant'."""
    t = UnitTracker(PEKKA_OURS)
    t.played("pekka", 0.0, (8, 0))
    tracks = t.update([det("giant", 8, 5)], 2.0)
    assert tracks[0].side == "shadow" and t.spawned == []

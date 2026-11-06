"""Life left of an enemy in our half, guessed from our towers' and troops' fire.

User, 2026-11-05: cards were sent at troops that had no life left.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from hp_estimate import NEARLY_DEAD_SHARE, TOWER_DPS, HpEstimator  # noqa: E402
from unit_stats import unit_hp, unit_value  # noqa: E402

BOTH_TOWERS = {"left": True, "right": True}


def track(name, x, y, id=1):
    return SimpleNamespace(id=id, name=name, x=x, y=y)


def run(estimator, enemies, allies, seconds, towers=BOTH_TOWERS, step=0.1):
    now = 0.0
    estimator.update(enemies, allies, towers, now)
    while now < seconds - 1e-9:
        now += step
        estimator.update(enemies, allies, towers, now)


def test_unit_stats_know_how_many_units_a_card_brings():
    assert abs(unit_value("skeleton") - 1 / 3) < 1e-6
    assert unit_value("knight") == 3
    assert unit_hp("knight") and unit_hp("knight") > unit_hp("skeleton")


def test_a_musketeer_our_tower_shoots_for_long_is_nearly_dead():
    hp = HpEstimator()
    musketeer = track("musketeer", 4, 9)
    run(hp, [musketeer], [], seconds=unit_hp("musketeer") / TOWER_DPS * 0.8)
    assert hp.left(musketeer) < NEARLY_DEAD_SHARE


def test_out_of_the_towers_reach_nothing_is_lost():
    hp = HpEstimator()
    giant = track("giant", 9, 14)
    run(hp, [giant], [], seconds=5)
    assert hp.left(giant) == 1.0


def test_a_fallen_tower_shoots_no_more():
    hp = HpEstimator()
    knight = track("knight", 4, 8)
    run(hp, [knight], [], seconds=5, towers={"left": False, "right": True})
    assert hp.left(knight) == 1.0


def test_the_tower_shoots_the_nearest_enemy_only():
    hp = HpEstimator()
    near, far = track("knight", 4, 8, id=1), track("knight", 5, 12, id=2)
    run(hp, [near, far], [], seconds=3)
    assert hp.left(near) < 1.0 and hp.left(far) == 1.0


def test_our_troops_on_it_count_but_not_against_a_flier_they_cannot_hit():
    hp = HpEstimator(flies=lambda name: name == "baby_dragon",
                     hits_air=lambda name: name != "knight")
    dragon, valkyrie = track("baby_dragon", 9, 13, id=1), track("valkyrie", 9, 13.5, id=2)
    run(hp, [dragon, valkyrie], [track("knight", 9, 12.5, id=9)], seconds=3)
    assert hp.left(dragon) == 1.0 and hp.left(valkyrie) < 1.0


def test_an_unknown_unit_is_always_at_full_life():
    hp = HpEstimator()
    odd = track("not_a_unit", 4, 8)
    run(hp, [odd], [], seconds=10)
    assert hp.left(odd) == 1.0


def test_reset_forgets_the_damage():
    hp = HpEstimator()
    knight = track("knight", 4, 8)
    run(hp, [knight], [], seconds=5)
    hp.reset()
    assert hp.left(knight) == 1.0

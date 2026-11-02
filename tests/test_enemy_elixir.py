"""The opponent's elixir: regeneration, and charges for what they deploy."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "vendor" / "ClashRoyaleBuildABot")):
    if path not in sys.path:
        sys.path.insert(0, path)

from enemy_elixir import EnemyElixir, card_cost  # noqa: E402


def started(at=0.0):
    tracker = EnemyElixir()
    tracker.start(at)
    return tracker


def test_starts_at_five_regenerates_and_caps_at_ten():
    t = started()
    assert t.value_at(0.0) == 5.0
    assert abs(t.value_at(2.8) - 6.0) < 0.01
    assert t.value_at(60.0) == 10.0


def test_phases_double_at_two_minutes_and_triple_at_three():
    t = started()
    assert t.multiplier(10.0) == 1.0
    assert t.multiplier(126.0) == 2.0
    assert t.multiplier(185.0) == 2.0      # overtime starts at 2x
    assert t.multiplier(245.0) == 3.0      # triple from 4:00, as the simulator


def test_regeneration_is_twice_as_fast_in_double_elixir():
    t = started()
    t.value_at(125.0)
    t.charge("giant", 125.0)
    before = t.value_at(125.0)
    assert abs((t.value_at(126.4) - before) - 1.0) < 0.05


def test_a_deploy_is_charged_its_card_cost():
    t = started()
    t.value_at(20.0)                                   # full
    assert t.charge("giant", 20.0) == 5
    assert abs(t.value_at(20.0) - 5.0) < 0.01


def test_a_multi_unit_card_is_charged_once():
    t = started()
    t.value_at(20.0)
    charged = sum(t.charge("minion", 20.0 + i * 0.05) for i in range(3))
    assert charged == 3                                # one Minions card, 3 elixir


def test_the_same_card_later_is_charged_again():
    t = started()
    t.value_at(30.0)
    t.charge("knight", 30.0)
    assert t.charge("knight", 34.0) == 3


def test_costs_come_from_the_card_table():
    assert card_cost("giant") == 5
    assert card_cost("minion") == 3
    assert card_cost("archer") == 3
    assert card_cost("goblin_cage") == 4
    assert card_cost("fireball") == 4
    assert card_cost("the_log") == 2
    assert card_cost("not_a_card") == 0


def test_never_below_zero():
    t = started()
    t.charge("giant", 0.0)
    t.charge("musketeer", 0.0)
    assert t.value_at(0.0) == 0.0

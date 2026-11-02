"""The user's rules for the P.E.K.K.A deck (2026-11-01).

Arrows only on small troops; P.E.K.K.A behind the King Tower for a big push,
never on defence; Musketeer / Minions / Archers behind it first; the Knight
only on defence; no troops at the bridge without a push; big pushes first.
Tiles are BuildABot's: y 0 is our back row, 14 the river on our side.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from pekka_strategy import PekkaStrategy  # noqa: E402

COSTS = {"pekka": 7, "musketeer": 4, "archers": 3, "minions": 3, "minipekka": 4,
         "knight": 3, "arrows": 3, "inferno_tower": 5}


def card(name):
    return SimpleNamespace(name=name, cost=COSTS[name])


def unit(name, x, y, transport="ground", target="ground"):
    return SimpleNamespace(unit=SimpleNamespace(name=name, transport=transport, target=target),
                           position=SimpleNamespace(tile_x=x, tile_y=y, conf=0.9))


def state(hand, elixir, allies=(), enemies=(), ready=None):
    number = SimpleNamespace
    return SimpleNamespace(
        cards=[card("knight"), *[card(n) for n in hand]],
        ready=list(range(len(hand))) if ready is None else ready,
        allies=list(allies), enemies=list(enemies),
        numbers=SimpleNamespace(elixir=number(number=elixir),
                                left_enemy_princess_hp=number(number=1.0),
                                right_enemy_princess_hp=number(number=1.0)))


def test_the_pekka_goes_behind_the_king_tower_at_full_elixir():
    plan = PekkaStrategy().decide(state(["pekka", "knight", "archers", "arrows"], 10), 0.0)
    assert plan.card == "pekka" and plan.tile in {(8, 0), (9, 0)}


def test_below_full_elixir_it_banks_for_the_pekka():
    plan = PekkaStrategy().decide(state(["pekka", "knight", "archers", "arrows"], 7), 0.0)
    assert plan.card is None and "P.E.K.K.A" in plan.reason


def test_support_behind_the_pekka_is_musketeer_then_minions_then_archers_never_the_knight():
    strategy = PekkaStrategy()
    strategy.played("pekka", (8, 0), now=0.0)
    walking = [unit("pekka", 4, 7)]
    plan = strategy.decide(state(["knight", "archers", "musketeer", "minions"], 7, allies=walking), 9.0)
    assert plan.card == "musketeer"
    plan = strategy.decide(state(["knight", "archers", "minions", "arrows"], 7, allies=walking), 9.5)
    assert plan.card == "minions"
    plan = strategy.decide(state(["knight", "arrows", "minipekka", "inferno_tower"], 9,
                                 allies=walking), 10.0)
    assert plan.card != "knight"


def test_the_pekka_is_never_a_defender():
    plan = PekkaStrategy().decide(
        state(["pekka", "arrows", "inferno_tower", "musketeer"], 10,
              enemies=[unit("valkyrie", 4, 8)], ready=[0, 1]), 0.0)
    assert plan.card != "pekka"


def test_a_tank_crossing_is_met_by_the_inferno_tower_in_the_middle():
    plan = PekkaStrategy().decide(
        state(["inferno_tower", "knight", "archers", "pekka"], 7, enemies=[unit("giant", 4, 9)]), 0.0)
    assert plan.card == "inferno_tower"
    assert 7 <= plan.tile[0] <= 10


def test_the_knight_defends():
    plan = PekkaStrategy().decide(
        state(["knight", "pekka", "arrows", "inferno_tower"], 4, enemies=[unit("valkyrie", 4, 8)]), 0.0)
    assert plan.card == "knight"


def test_arrows_go_on_a_group_of_small_troops():
    goblins = [unit("goblin", 5, 9), unit("goblin", 6, 9), unit("spear_goblin", 5, 10)]
    plan = PekkaStrategy().decide(state(["arrows", "pekka", "knight", "archers"], 5, enemies=goblins), 0.0)
    assert plan.card == "arrows"


def test_arrows_never_go_on_a_big_troop():
    plan = PekkaStrategy().decide(
        state(["arrows", "pekka", "musketeer", "archers"], 5,
              enemies=[unit("giant", 5, 20), unit("musketeer", 6, 21)]), 0.0)
    assert plan.card != "arrows"


def test_it_casts_its_own_spells():
    assert PekkaStrategy().handles_spells is True


def test_full_elixir_without_the_pekka_cycles_a_cheap_card_in_the_corner_not_the_knight():
    strategy = PekkaStrategy()
    hand = ["knight", "archers", "minions", "arrows"]
    strategy.decide(state(hand, 10), 0.0)
    plan = strategy.decide(state(hand, 10), 3.0)
    assert plan.card in {"archers", "minions"}
    assert plan.tile in {(3, 1), (14, 1)}


def test_a_musketeer_stays_well_back_from_a_mini_pekka():
    plan = PekkaStrategy().decide(state(["musketeer", "pekka", "arrows", "archers"], 5,
                                        enemies=[unit("minipekka", 4, 9)], ready=[0]), 0.0)
    assert plan.card == "musketeer" and plan.tile[1] <= 3


def test_the_knight_goes_in_the_middle():
    plan = PekkaStrategy().decide(
        state(["knight", "pekka", "arrows", "inferno_tower"], 4, enemies=[unit("valkyrie", 3, 9)]), 0.0)
    assert plan.card == "knight" and plan.tile[0] in (8, 9)


def test_the_inferno_tower_goes_up_as_the_tank_reaches_the_bridge():
    plan = PekkaStrategy().decide(
        state(["inferno_tower", "knight", "archers", "pekka"], 6, enemies=[unit("giant", 4, 18)]), 0.0)
    assert plan.card == "inferno_tower" and plan.tile[1] == 10

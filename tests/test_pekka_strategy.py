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


# --- pekka2: a counter per threat (live 2026-11-04: 1 win in 8 vs a Wizard) --


def counters_strategy():
    from pekka_strategy import PekkaCountersStrategy
    return PekkaCountersStrategy()


def test_pekka2_answers_a_wizard_with_the_mini_pekka_not_archers():
    hand = ["archers", "minions", "minipekka", "arrows"]
    plan = counters_strategy().decide(state(hand, 7, enemies=[unit("wizard", 4, 9)]), 0.0)
    assert plan.card == "minipekka"


def test_the_plain_pekka_rules_keep_their_generic_order():
    hand = ["archers", "minions", "minipekka", "arrows"]
    plan = PekkaStrategy().decide(state(hand, 7, enemies=[unit("wizard", 4, 9)]), 0.0)
    assert plan.card == "archers"


def test_pekka2_sends_minions_on_a_valkyrie():
    hand = ["knight", "minions", "minipekka", "arrows"]
    plan = counters_strategy().decide(state(hand, 7, enemies=[unit("valkyrie", 4, 9)]), 0.0)
    assert plan.card == "minions"


def test_pekka2_keeps_minions_off_a_baby_dragon():
    hand = ["minions", "musketeer", "knight", "arrows"]
    plan = counters_strategy().decide(
        state(hand, 7, enemies=[unit("baby_dragon", 4, 9, transport="air", target="all")]), 0.0)
    assert plan.card == "musketeer"


def test_pekka2_puts_the_inferno_tower_on_a_balloon():
    hand = ["musketeer", "inferno_tower", "minions", "arrows"]
    plan = counters_strategy().decide(
        state(hand, 7, enemies=[unit("balloon", 4, 12, transport="air", target="buildings")]), 0.0)
    assert plan.card == "inferno_tower"


def test_pekka2_falls_back_on_the_generic_list_when_no_counter_is_ready():
    hand = ["archers", "minions", "arrows", "pekka"]
    plan = counters_strategy().decide(state(hand, 7, enemies=[unit("wizard", 4, 9)]), 0.0)
    assert plan.card == "archers"


def test_a_building_read_as_an_enemy_in_our_half_is_not_defended_against():
    hand = ["knight", "minions", "minipekka", "arrows"]
    plan = PekkaStrategy().decide(state(hand, 7, enemies=[unit("inferno_tower", 9, 9)]), 0.0)
    assert plan.card is None or plan.phase != "défense"


# --- user, 2026-11-05: the Inferno Tower still came too late -------------------


def test_a_hog_rider_on_its_way_gets_the_inferno_tower_before_it_crosses():
    hand = ["inferno_tower", "knight", "minipekka", "arrows"]
    plan = PekkaStrategy().decide(state(hand, 6, enemies=[unit("hog_rider", 14, 21)]), 0.0)
    assert plan.card == "inferno_tower"


def test_a_tank_still_on_their_side_gets_no_troop_and_elixir_is_kept():
    hand = ["knight", "minipekka", "archers", "arrows"]
    plan = PekkaStrategy().decide(state(hand, 6, enemies=[unit("giant", 4, 21)]), 0.0)
    assert plan.card is None


def test_a_balloon_gets_the_inferno_tower_first():
    hand = ["musketeer", "inferno_tower", "minions", "arrows"]
    plan = PekkaStrategy().decide(
        state(hand, 7, enemies=[unit("balloon", 4, 12, transport="air", target="buildings")]), 0.0)
    assert plan.card == "inferno_tower"


def test_an_enemy_at_our_tower_comes_before_a_tank_still_on_its_way():
    """Live 2026-11-05: a Mega Knight at their bridge held the elixir while a
    Bomber shot our tower."""
    hand = ["minions", "knight", "archers", "inferno_tower"]
    plan = PekkaStrategy().decide(state(hand, 7, enemies=[unit("mega_knight", 4, 20),
                                                          unit("bomber", 3, 7)]), 0.0)
    assert plan.card is not None and plan.tile[1] < 12


def test_a_tank_read_right_beside_our_own_troop_on_their_side_is_ignored():
    hand = ["minions", "knight", "archers", "arrows"]
    plan = PekkaStrategy().decide(state(hand, 7, allies=[unit("minipekka", 4, 22)],
                                        enemies=[unit("pekka", 4, 22)]), 0.0)
    assert "il arrive" not in plan.reason


def test_without_the_inferno_tower_in_hand_a_full_bar_is_not_held_for_it():
    hand = ["arrows", "knight", "archers", "minions"]
    plan = PekkaStrategy().decide(state(hand, 10, enemies=[unit("giant", 3, 18)]), 0.0)
    assert "garde l'élixir" not in plan.reason


def test_the_pekka_still_does_not_go_out_to_meet_a_tank_on_its_way():
    hand = ["pekka", "knight", "archers", "arrows"]
    plan = PekkaStrategy().decide(state(hand, 10, enemies=[unit("giant", 3, 18)]), 0.0)
    assert plan.card != "pekka"


# --- live 2026-11-05: 17 pushes out of 17 went left (both towers read 1.0) ---


def test_with_equal_towers_the_push_goes_where_our_troops_already_are():
    hand = ["pekka", "knight", "archers", "arrows"]
    plan = PekkaStrategy().decide(state(hand, 10, allies=[unit("minipekka", 14, 12)]), 0.0)
    assert plan.card == "pekka" and plan.tile == (9, 0)


def test_with_equal_towers_and_no_troops_the_next_push_takes_the_other_lane():
    strategy = PekkaStrategy()
    hand = ["pekka", "knight", "archers", "arrows"]
    first = strategy.decide(state(hand, 10), 0.0)
    strategy.played(first.card, first.tile, 0.5)
    strategy.push = None                                    # that push is over
    second = strategy.decide(state(hand, 10), 60.0)
    assert {first.tile, second.tile} == {(8, 0), (9, 0)}


def test_a_weaker_enemy_tower_still_decides_the_lane():
    hand = ["pekka", "knight", "archers", "arrows"]
    s = state(hand, 10, allies=[unit("minipekka", 3, 12)])
    s.numbers.right_enemy_princess_hp.number = 0.6
    plan = PekkaStrategy().decide(s, 0.0)
    assert plan.tile == (9, 0)


# --- pekka2c: pekka2 with the 2026-11-05 defence additions off, for an A/B ---


def classic_strategy():
    from pekka_strategy import PekkaClassicStrategy
    return PekkaClassicStrategy()


def test_pekka2c_still_answers_a_troop_the_life_estimate_calls_dying():
    from types import SimpleNamespace as NS
    dying = NS(unit=NS(name="valkyrie", transport="ground", target="ground"),
               position=NS(tile_x=4, tile_y=8, conf=0.9), hp_left=0.1)
    plan = classic_strategy().decide(state(["knight", "minions", "archers", "arrows"], 8,
                                           enemies=[dying]), 0.0)
    assert plan.phase == "défense" and plan.card is not None


def test_pekka2c_has_no_price_check():
    skeletons = [unit("skeleton", 13, 12), unit("skeleton", 14, 12), unit("skeleton", 13, 11)]
    plan = classic_strategy().decide(state(["archers", "knight", "minions", "musketeer"], 8,
                                           enemies=skeletons), 0.0)
    assert plan.phase == "défense" and plan.card is not None


def test_pekka2c_alerts_tanks_from_row_19_only():
    hand = ["inferno_tower", "knight", "minipekka", "arrows"]
    assert classic_strategy().decide(state(hand, 6, enemies=[unit("giant", 4, 21)]), 0.0).card is None
    plan = classic_strategy().decide(state(hand, 6, enemies=[unit("giant", 4, 18)]), 0.0)
    assert plan.card == "inferno_tower"


def test_pekka2c_asks_the_runner_for_one_frame_enemies():
    assert classic_strategy().enemy_min_hits == 1
    assert getattr(PekkaStrategy(), "enemy_min_hits", None) is None


# --- pekka_hier: the rules as they were on 2026-11-04, for an A/B --------------


def yesterday_strategy():
    from pekka_strategy import PekkaYesterdayStrategy
    return PekkaYesterdayStrategy()


def test_pekka_hier_pushes_left_on_equal_towers_and_keeps_yesterdays_switches():
    s = yesterday_strategy()
    plan = s.decide(state(["pekka", "knight", "archers", "arrows"], 10,
                          allies=[unit("minipekka", 14, 12)]), 0.0)
    assert plan.tile == (8, 0)
    assert not (s.gap_per_lane or s.lane_by_troops or s.ignore_structures or s.own_structures)
    assert s.counters is None


def test_a_knight_dropped_in_the_middle_covers_a_threat_by_the_edge():
    """Live 2026-11-05: a Knight at column 8 did not count for a threat at x=1."""
    strategy = PekkaStrategy()
    hand = ["knight", "minipekka", "archers", "arrows"]
    first = strategy.decide(state(hand, 9, enemies=[unit("valkyrie", 1, 9)]), 0.0)
    assert first.card == "knight" and 7 <= first.tile[0] <= 10
    strategy.played(first.card, first.tile, 0.2)
    plan = strategy.decide(state(["minipekka", "archers", "minions", "arrows"], 6,
                                 enemies=[unit("valkyrie", 1, 8)]), 1.0)
    assert plan.card is None or plan.phase != "défense"

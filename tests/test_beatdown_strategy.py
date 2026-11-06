"""The 'gros push' plan for the starter deck: Giant at the back, support behind it.

Tiles are BuildABot's: x 0-17 left to right, y 0 at our back row up to 14 at
the river on our side, enemy units across the river read y >= 17.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from beatdown_strategy import BeatdownStrategy  # noqa: E402

COSTS = {"giant": 5, "musketeer": 4, "archers": 3, "minions": 3, "minipekka": 4,
         "knight": 3, "fireball": 4, "arrows": 3}


def card(name):
    return SimpleNamespace(name=name, cost=COSTS[name])


def unit(name, x, y, conf=0.9, transport="ground", target="ground"):
    return SimpleNamespace(unit=SimpleNamespace(name=name, transport=transport, target=target),
                           position=SimpleNamespace(tile_x=x, tile_y=y, conf=conf))


def state(hand, elixir, allies=(), enemies=(), left_hp=1.0, right_hp=1.0, ready=None):
    number = SimpleNamespace
    return SimpleNamespace(
        cards=[card("knight"), *[card(n) for n in hand]],
        ready=list(range(len(hand))) if ready is None else ready,
        allies=list(allies), enemies=list(enemies),
        numbers=SimpleNamespace(
            elixir=number(number=elixir),
            left_enemy_princess_hp=number(number=left_hp),
            right_enemy_princess_hp=number(number=right_hp)))


def test_saves_elixir_for_the_giant_and_says_so():
    plan = BeatdownStrategy().decide(state(["giant", "knight", "archers", "arrows"], 6), 0.0)
    assert plan.card is None and not plan.delegate
    assert "Géant" in plan.reason and "6/10" in plan.reason


def test_giant_goes_behind_the_king_tower_on_the_weaker_tower_side():
    strategy = BeatdownStrategy()
    plan = strategy.decide(state(["knight", "giant", "archers", "arrows"], 10, right_hp=0.4), 0.0)
    assert plan.card == "giant" and plan.slot == 1
    assert plan.tile == (9, 0)                        # right side of the King Tower
    plan = strategy.decide(state(["giant", "knight", "archers", "arrows"], 10, left_hp=0.3), 0.0)
    assert plan.tile == (8, 0)


def test_support_is_placed_behind_the_walking_giant():
    strategy = BeatdownStrategy()
    strategy.played("giant", (8, 0), now=0.0)
    hand = ["musketeer", "knight", "archers", "arrows"]
    plan = strategy.decide(state(hand, 5, allies=[unit("giant", 4, 7)]), 8.0)
    assert plan.card == "musketeer"
    assert plan.tile == (4, 5)                        # two tiles behind it, same column
    assert "derrière le Géant" in plan.reason


def test_support_waits_until_the_giant_has_room_behind_it():
    strategy = BeatdownStrategy()
    strategy.played("giant", (8, 0), now=0.0)
    plan = strategy.decide(state(["musketeer", "knight", "archers", "arrows"], 6,
                                 allies=[unit("giant", 8, 1)]), 1.0)
    assert plan.card is None


def test_support_at_the_bridge_foot_once_the_giant_has_crossed():
    strategy = BeatdownStrategy()
    strategy.played("giant", (9, 0), now=0.0)
    plan = strategy.decide(state(["archers", "knight", "minions", "arrows"], 6,
                                 allies=[unit("giant", 14, 19)]), 18.0)
    assert plan.card == "archers"
    assert plan.tile == (14, 14)


def test_push_ends_when_the_giant_is_gone():
    strategy = BeatdownStrategy()
    strategy.played("giant", (8, 0), now=0.0)
    strategy.decide(state(["knight", "archers", "minions", "arrows"], 3,
                          allies=[unit("giant", 3, 9)]), 10.0)
    plan = strategy.decide(state(["knight", "archers", "minions", "arrows"], 3), 20.0)
    assert plan.phase == "économie"


def test_a_ground_threat_is_met_in_its_lane_by_a_melee_card_not_the_giant():
    """BuildABot's own rules dropped the Knight mid-field at (8,9) and, at 10
    elixir, Archers or Minions anywhere: troops up the middle with no Giant."""
    plan = BeatdownStrategy().decide(
        state(["giant", "knight", "archers", "arrows"], 8, enemies=[unit("hog_rider", 14, 9)]), 0.0)
    assert plan.phase == "défense" and not plan.delegate
    assert plan.card == "knight"
    assert plan.tile == (14, 7)                     # in front of it, in its lane
    assert "hog_rider" in plan.reason


def test_an_air_threat_gets_something_that_hits_air_behind_it():
    plan = BeatdownStrategy().decide(
        state(["knight", "archers", "giant", "arrows"], 6,
              enemies=[unit("baby_dragon", 5, 10, transport="air")]), 0.0)
    assert plan.card == "archers"
    x, y = plan.tile
    assert y < 10 and 5 <= x <= 8                   # behind it, toward the King, out of its path


def test_minions_go_straight_onto_the_threat():
    trio = [unit("minion", 4, 11, transport="air"), unit("minion", 5, 11, transport="air"),
            unit("minion", 4, 12, transport="air")]
    plan = BeatdownStrategy().decide(
        state(["minions", "giant", "arrows", "fireball"], 6, enemies=trio), 0.0)
    assert plan.card == "minions" and plan.tile == (4, 11)


def test_one_defender_per_wave_not_the_whole_hand():
    strategy = BeatdownStrategy()
    hand = ["knight", "archers", "giant", "arrows"]
    threat = [unit("knight", 4, 10)]
    first = strategy.decide(state(hand, 8, enemies=threat), 0.0)
    strategy.played(first.card, first.tile, 0.0)
    again = strategy.decide(state(hand, 8, enemies=threat), 1.0)
    assert again.card is None                       # back to saving for the Giant
    assert "déjà posé" in again.reason


def test_enemies_on_their_own_side_are_not_a_threat():
    plan = BeatdownStrategy().decide(
        state(["knight", "archers", "giant", "arrows"], 6, enemies=[unit("knight", 8, 22)]), 0.0)
    assert plan.phase != "défense" and plan.card is None


def test_no_affordable_defender_says_so_and_never_delegates():
    plan = BeatdownStrategy().decide(
        state(["knight", "archers", "giant", "arrows"], 2, enemies=[unit("knight", 4, 10)]), 0.0)
    assert plan.card is None and not plan.delegate
    assert plan.phase == "défense" and "élixir" in plan.reason


def test_without_a_giant_it_waits_instead_of_sending_troops():
    strategy = BeatdownStrategy()
    plan = strategy.decide(state(["knight", "archers", "arrows", "fireball"], 10), 0.0)
    assert plan.card is None and plan.phase == "économie"
    assert "Géant" in plan.reason


def test_full_elixir_too_long_cycles_one_cheap_card_in_the_back_corner():
    strategy = BeatdownStrategy()
    strategy.decide(state(["knight", "archers", "arrows", "fireball"], 10), 0.0)
    plan = strategy.decide(state(["knight", "archers", "arrows", "fireball"], 10), 3.0)
    assert plan.card in ("knight", "archers")
    assert plan.tile in ((3, 1), (14, 1))          # never the middle
    assert plan.phase == "cycle"


def test_support_follows_the_giants_lane_when_vision_has_not_seen_it():
    strategy = BeatdownStrategy()
    strategy.played("giant", (8, 0), now=0.0)      # left of the King: walks to the left lane
    plan = strategy.decide(state(["musketeer", "knight", "archers", "arrows"], 6), 10.0)
    assert plan.card == "musketeer"
    assert plan.tile[0] <= 4                        # in the left lane, not the middle


def test_unaffordable_or_not_ready_cards_are_never_planned():
    plan = BeatdownStrategy().decide(state(["giant", "knight", "archers", "arrows"], 10, ready=[1, 2, 3]), 0.0)
    assert plan.card != "giant"


def test_a_lone_skeleton_is_left_to_the_towers():
    """The last match: a 4-elixir Mini P.E.K.K.A on every lone Skeleton and
    Goblin, and the Giant played once in three minutes."""
    plan = BeatdownStrategy().decide(
        state(["minipekka", "knight", "giant", "arrows"], 7, enemies=[unit("skeleton", 13, 14)]),
        0.0)
    assert plan.card is None
    assert "tours" in plan.reason


def test_a_swarm_of_small_units_is_answered_cheaply():
    goblins = [unit("goblin", 13, 13), unit("goblin", 14, 13), unit("goblin", 13, 12)]
    plan = BeatdownStrategy().decide(
        state(["minipekka", "knight", "giant", "arrows"], 8, enemies=goblins), 0.0)
    assert plan.card == "knight"


def test_a_tower_rusher_gets_the_mini_pekka():
    plan = BeatdownStrategy().decide(
        state(["knight", "minipekka", "giant", "arrows"], 8,
              enemies=[unit("hog_rider", 14, 12, target="buildings")]), 0.0)
    assert plan.card == "minipekka"


def test_a_lone_fighter_gets_the_cheapest_answer_not_the_mini_pekka():
    plan = BeatdownStrategy().decide(
        state(["minipekka", "knight", "giant", "arrows"], 8, enemies=[unit("valkyrie", 4, 11)]),
        0.0)
    assert plan.card == "knight"


def test_an_air_tank_gets_the_musketeer_first():
    plan = BeatdownStrategy().decide(
        state(["archers", "musketeer", "giant", "arrows"], 8,
              enemies=[unit("balloon", 14, 12, transport="air", target="buildings")]), 0.0)
    assert plan.card == "musketeer"


def test_the_most_dangerous_threat_is_answered_first():
    """A Hog Rider behind a lone Skeleton: the Hog is the one to stop."""
    enemies = [unit("skeleton", 4, 9), unit("hog_rider", 14, 13, target="buildings")]
    plan = BeatdownStrategy().decide(
        state(["knight", "minipekka", "giant", "arrows"], 8, enemies=enemies), 0.0)
    assert plan.card == "minipekka" and plan.tile[0] >= 12


def test_a_threat_already_fought_by_one_of_ours_gets_no_second_card():
    plan = BeatdownStrategy().decide(
        state(["minipekka", "knight", "giant", "arrows"], 8,
              allies=[unit("knight", 4, 9)], enemies=[unit("valkyrie", 4, 11)]), 10.0)
    assert plan.card is None
    assert "déjà" in plan.reason


def test_our_giant_walking_past_a_threat_is_not_a_defender():
    plan = BeatdownStrategy().decide(
        state(["minipekka", "knight", "giant", "arrows"], 8,
              allies=[unit("giant", 4, 10)], enemies=[unit("valkyrie", 4, 11)]), 10.0)
    assert plan.card == "knight"


def test_an_enemy_giant_is_a_tank_for_the_mini_pekka():
    """BuildABot lists the Giant as targeting ground, but it walks to towers."""
    plan = BeatdownStrategy().decide(
        state(["knight", "minipekka", "giant", "arrows"], 8, enemies=[unit("giant", 4, 12)]), 0.0)
    assert plan.card == "minipekka"


def test_a_knight_nearby_does_not_cover_a_balloon():
    plan = BeatdownStrategy().decide(
        state(["knight", "archers", "giant", "arrows"], 8, allies=[unit("knight", 14, 11)],
              enemies=[unit("balloon", 14, 12, transport="air", target="buildings")]), 0.0)
    assert plan.card == "archers"


def test_while_a_defender_works_the_giant_still_goes_at_full_elixir():
    plan = BeatdownStrategy().decide(
        state(["giant", "knight", "archers", "arrows"], 10,
              allies=[unit("knight", 4, 9)], enemies=[unit("valkyrie", 4, 11)]), 10.0)
    assert plan.card == "giant"


def test_a_hog_rider_is_answered_even_next_to_one_of_our_troops():
    """A Hog runs past troops to the tower: a Knight standing near it is not
    stopping it (seen live: the bot sent a Musketeer to the push instead)."""
    plan = BeatdownStrategy().decide(
        state(["knight", "minipekka", "giant", "arrows"], 8, allies=[unit("knight", 4, 7)],
              enemies=[unit("hog_rider", 4, 5, target="buildings")]), 10.0)
    assert plan.card == "minipekka"


def test_a_giant_gets_a_defence_budget_not_the_whole_hand():
    """Live: Knight, Archers, Knight, Minions, Mini P.E.K.K.A - 16 elixir on
    one 5-elixir Giant, a card every 3 s while it walked."""
    strategy = BeatdownStrategy()
    hand = ["knight", "minipekka", "archers", "minions"]
    giant = lambda y: [unit("giant", 4, y, target="buildings")]          # noqa: E731
    first = strategy.decide(state(hand, 10, enemies=giant(12)), 0.0)
    strategy.played(first.card, first.tile, 0.0)                         # Mini P.E.K.K.A: 4
    after = ["knight", "archers", "minions", "giant"]                      # Mini P.E.K.K.A played
    second = strategy.decide(state(after, 10, enemies=giant(10)), 3.5)
    strategy.played(second.card, second.tile, 3.5)                       # Knight: 3
    third = strategy.decide(state(["archers", "minions", "giant", "arrows"], 10,
                                  enemies=giant(8)), 7.0)
    assert (first.card, second.card) == ("minipekka", "knight")
    assert third.card is None or third.phase != "défense"
    assert "engagé" in third.reason


def test_the_budget_frees_up_once_the_wave_is_over():
    strategy = BeatdownStrategy()
    hand = ["knight", "minipekka", "archers", "minions"]
    for t in (0.0, 3.5):
        plan = strategy.decide(state(hand, 10, enemies=[unit("giant", 4, 10, target="buildings")]), t)
        strategy.played(plan.card, plan.tile, t)
    later = strategy.decide(state(hand, 10, enemies=[unit("hog_rider", 4, 10, target="buildings")]), 30.0)
    assert later.card == "minipekka"


# --- user, 2026-11-05: useless defenders (a troop with no life left, Archers
# for two Skeletons) ----------------------------------------------------------


def test_two_skeletons_are_not_worth_a_card():
    skeletons = [unit("skeleton", 13, 12), unit("skeleton", 14, 12), unit("skeleton", 13, 11)]
    plan = BeatdownStrategy().decide(
        state(["archers", "knight", "giant", "arrows"], 8, enemies=skeletons), 0.0)
    assert plan.card is None or plan.phase != "défense"


def test_a_troop_our_tower_has_all_but_killed_gets_no_card():
    from types import SimpleNamespace as NS
    dying = NS(unit=NS(name="valkyrie", transport="ground", target="ground"),
               position=NS(tile_x=4, tile_y=8, conf=0.9), hp_left=0.1)
    plan = BeatdownStrategy().decide(state(["knight", "archers", "giant", "arrows"], 8,
                                           enemies=[dying]), 0.0)
    assert plan.card is None or plan.phase != "défense"


def test_a_defender_in_one_lane_does_not_delay_the_answer_in_the_other():
    strategy = BeatdownStrategy()
    first = strategy.decide(state(["knight", "minipekka", "giant", "arrows"], 9,
                                  enemies=[unit("valkyrie", 3, 9)]), 0.0)
    strategy.played(first.card, first.tile, 0.2)
    plan = strategy.decide(state(["minipekka", "archers", "giant", "arrows"], 6,
                                 enemies=[unit("valkyrie", 3, 9), unit("hog_rider", 14, 12)]), 1.0)
    assert plan.phase == "défense" and plan.card == "minipekka" and plan.tile[0] >= 10

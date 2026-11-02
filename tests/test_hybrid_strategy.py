"""The trained policy attacks, the rule engine's defence answers what crosses.

Measured on 116 live matches (2026-10-27): the policy loses when enemy units
reach our towers (twice as often in losses) and answers an enemy entering
our half after a median 4 s, having spent its elixir at the bridge. The
beatdown rules already defend: assess the threat, skip it when engaged or
paid for, pick a defender and its tile.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from beatdown_strategy import DEFENCE, Plan  # noqa: E402
from hybrid_strategy import HybridStrategy  # noqa: E402

COSTS = {"giant": 5, "musketeer": 4, "archers": 3, "minions": 3, "minipekka": 4,
         "knight": 3, "fireball": 4, "arrows": 3}


def card(name):
    return SimpleNamespace(name=name, cost=COSTS[name])


def unit(name, x, y, transport="ground", target="ground"):
    return SimpleNamespace(unit=SimpleNamespace(name=name, transport=transport, target=target),
                           position=SimpleNamespace(tile_x=x, tile_y=y, conf=0.9))


def state(hand, elixir, enemies=(), allies=(), ready=None):
    number = SimpleNamespace
    return SimpleNamespace(
        cards=[card("fireball"), *[card(n) for n in hand]],
        ready=list(range(len(hand))) if ready is None else ready,
        allies=list(allies), enemies=list(enemies),
        numbers=SimpleNamespace(elixir=number(number=elixir),
                                left_enemy_princess_hp=number(number=1.0),
                                right_enemy_princess_hp=number(number=1.0)))


class FakeIA:
    name = "ia"
    handles_spells = True

    def __init__(self, card="archers"):
        self.card, self.played_cards, self.started = card, [], None

    def start_match(self, now):
        self.started = now

    def played(self, card, tile, now):
        self.played_cards.append(card)

    def decide(self, state, now):
        return Plan(phase="ia", card=self.card, slot=0, tile=(3, 13), reason="IA : au pont")


HAND = ["archers", "knight", "minipekka", "arrows"]


def test_without_a_threat_the_policy_plays():
    plan = HybridStrategy(FakeIA()).decide(state(HAND, 7), 0.0)
    assert plan.phase == "ia" and plan.tile == (3, 13)


def test_an_enemy_crossing_into_our_half_is_answered_by_the_rules():
    plan = HybridStrategy(FakeIA()).decide(state(HAND, 7, enemies=[unit("valkyrie", 4, 8)]), 0.0)
    assert plan.phase == DEFENCE
    assert plan.card in HAND and plan.tile != (3, 13)


def test_threatened_with_no_defender_ready_it_keeps_its_elixir():
    plan = HybridStrategy(FakeIA()).decide(
        state(HAND, 2, enemies=[unit("valkyrie", 4, 8)], ready=[]), 0.0)
    assert plan.card is None


def test_once_a_defender_is_down_the_policy_plays_again():
    hybrid = HybridStrategy(FakeIA())
    threatened = state(HAND, 7, enemies=[unit("valkyrie", 4, 8)])
    defence = hybrid.decide(threatened, 0.0)
    hybrid.played(defence.card, defence.tile, 0.5)
    assert hybrid.decide(threatened, 1.0).phase == "ia"


def test_the_policy_hears_every_play_and_every_new_match():
    ia = FakeIA()
    hybrid = HybridStrategy(ia)
    hybrid.start_match(3.0)
    hybrid.played("knight", (4, 6), 4.0)
    assert ia.started == 3.0 and ia.played_cards == ["knight"]


def test_cluster_spells_hand_the_runner_its_spell_rule_back():
    """The policy cast Fireball 0.2 and Arrows 0.4-0.7 times a match live:
    with cluster_spells the runner's spell-on-a-cluster rule plays first and
    the policy still decides everything else."""
    assert HybridStrategy(FakeIA()).handles_spells is True
    assert HybridStrategy(FakeIA(), cluster_spells=True).handles_spells is False


# ------------------------------------------------- the P.E.K.K.A deck (2026-11-02)

COSTS.update({"pekka": 7, "inferno_tower": 5, "skeleton": 1})
PEKKA_HAND = ["inferno_tower", "knight", "pekka", "arrows"]


def pekka_hybrid(ia_card="archers"):
    from pekka_strategy import PekkaStrategy
    return HybridStrategy(FakeIA(ia_card), rules_factory=PekkaStrategy)


def test_with_the_pekka_rules_a_tank_is_answered_before_it_crosses():
    """User: the Inferno Tower came too late. The P.E.K.K.A rules answer a
    tank as it reaches the bridge (row 19), not only once in our half."""
    plan = pekka_hybrid().decide(state(PEKKA_HAND, 7, enemies=[unit("giant", 4, 18)]), 0.0)
    assert plan.phase == DEFENCE and plan.card == "inferno_tower"


def test_with_the_pekka_rules_arrows_go_on_a_group_of_small_troops():
    small = [unit("skeleton", 4, 20), unit("skeleton", 5, 20), unit("skeleton", 4, 21)]
    plan = pekka_hybrid().decide(state(PEKKA_HAND, 7, enemies=small), 0.0)
    assert plan.card == "arrows"


def test_with_the_pekka_rules_the_policy_may_not_arrow_a_lone_big_unit():
    """User: Arrows only on small troops."""
    plan = pekka_hybrid("arrows").decide(state(PEKKA_HAND, 7, enemies=[unit("giant", 4, 25)]), 0.0)
    assert plan.card is None


def test_with_the_pekka_rules_the_policy_may_not_defend_with_the_pekka():
    """User: the P.E.K.K.A never goes in on defence. The defender is already
    down, so the policy's plan would pass through - not when it is the
    P.E.K.K.A dropped on the threat."""
    hybrid = pekka_hybrid("pekka")
    threatened = state(PEKKA_HAND, 9, enemies=[unit("valkyrie", 4, 8)])
    defence = hybrid.decide(threatened, 0.0)
    hybrid.played(defence.card, defence.tile, 0.5)
    plan = hybrid.decide(threatened, 1.0)
    assert plan.card != "pekka"


def test_with_the_pekka_rules_the_policy_still_attacks_when_nothing_threatens():
    plan = pekka_hybrid("pekka").decide(state(PEKKA_HAND, 9), 0.0)
    assert plan.phase == "ia" and plan.card == "pekka"

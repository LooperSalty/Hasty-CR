"""A simulator-trained policy driving the native runner, for any 8-card deck.

Two halves, each checked against the simulator the policy was trained in:

* the live bridge (`brain.rl_policy`) generalised from Hog 2.6 to the deck the
  checkpoint was trained on - same board, both routes, same observation and
  the same legal actions (Arrows reach the enemy half like any spell);
* `native_rl`, which builds that live observation from the native runner's
  BuildABot state (tiles, card names, tower bars) and turns the chosen action
  back into a card slot and a tile.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "vendor" / "ClashRoyaleBuildABot")):
    if path not in sys.path:
        sys.path.insert(0, path)

from brain.rl_policy import mask_from_live, observation_from_live  # noqa: E402
from sim import arena as sim_arena                                 # noqa: E402
from sim.env import TILES, ClashEnv, legal_mask, observe           # noqa: E402

STARTER = ["knight", "archer", "minions", "arrows", "fireball", "giant", "musketeer",
           "minipekka"]


def played_out(seed: int = 5, steps: int = 240):
    env = ClashEnv(seed=seed, opponent="meta", deck=STARTER)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        legal = np.flatnonzero(legal_mask(env.match, env._cards, 1))
        _obs, _r, term, trunc, _info = env.step(int(rng.choice(legal)))
        if term or trunc:
            break
    return env


def live_view_of(match):
    allies, tracks = [], []
    for entity in match.battle.entities.values():
        if not entity.alive or entity.is_tower:
            continue
        x, y = sim_arena.to_tiles(entity.pos)
        cell = SimpleNamespace(x=x, y=y)
        if entity.side > 0:
            allies.append((cell, entity.name))
        else:
            tracks.append(SimpleNamespace(cell=cell, name=entity.name))
    player = match.players[1]
    obs = SimpleNamespace(
        allies=[cell for cell, _ in allies], tracks=tracks,
        elixir=player.elixir / 1000.0, elapsed=match.elapsed_ms / 1000.0,
        multiplier=2800.0 / max(1, match.regen_ms()),
        ally_hp=match.tower_fractions(1), enemy_hp=match.tower_fractions(-1),
        hand={card: slot for slot, card in enumerate(player.hand[:4])},
        next_card=player.next_card)
    return obs, allies


# ------------------------------------------------ the bridge, any 8-card deck

def test_starter_deck_observation_matches_the_simulator():
    env = played_out()
    obs, allies = live_view_of(env.match)
    sim = observe(env.match, 1)
    live = observation_from_live(obs, allies, deck=STARTER)
    # Planes 2-3 are hit points: the simulator sees what is left, live vision
    # only a card's full HP - the documented gap (tests/test_rl_policy.py).
    for plane in (0, 1, 4, 5, 6, 7):
        assert np.allclose(sim["planes"][plane], live["planes"][plane], atol=1e-3), plane
    assert live["planes"][2].sum() > 0, "starter units must still get their HP"
    assert np.allclose(sim["scalars"], live["scalars"], atol=1e-3)


def test_starter_deck_mask_matches_the_simulator_spells_included():
    for seed in (5, 11, 17):
        env = played_out(seed=seed)
        obs, _ = live_view_of(env.match)
        sim = legal_mask(env.match, env._cards, 1)[:1 + 4 * TILES]
        live = mask_from_live(obs, deck=STARTER)[:1 + 4 * TILES]
        assert np.array_equal(sim, live), seed


def test_arrows_reach_the_enemy_half_live():
    obs = SimpleNamespace(elixir=10, hand={"arrows": 0}, enemy_hp={"left": 1, "right": 1})
    mask = mask_from_live(obs, deck=STARTER)
    enemy_row_cell = 1 + 0 * TILES + 5 * 18 + 8            # grid (8, 5): their half
    assert mask[enemy_row_cell]


# ------------------------------------------------- native state -> observation


class _Lazy:
    """`native_rl`, imported on first use so the bridge tests above run alone."""

    def __getattr__(self, name):
        import native_rl as module
        return getattr(module, name)


native_rl = _Lazy()


def card(name):
    return SimpleNamespace(name=name)


def det(name, x, y):
    return SimpleNamespace(unit=SimpleNamespace(name=name),
                           position=SimpleNamespace(tile_x=x, tile_y=y))


def native_state(hand=("knight", "archers", "giant", "arrows"), nxt="fireball", elixir=7,
                 allies=(), enemies=(), towers=(1.0, 1.0, 1.0, 1.0), ready=(0, 1, 2, 3)):
    number = SimpleNamespace
    ally_l, ally_r, enemy_l, enemy_r = towers
    return SimpleNamespace(
        cards=[card(nxt), *[card(n) for n in hand]], ready=list(ready),
        allies=list(allies), enemies=list(enemies),
        numbers=SimpleNamespace(
            elixir=number(number=elixir),
            left_ally_princess_hp=number(number=ally_l),
            right_ally_princess_hp=number(number=ally_r),
            left_enemy_princess_hp=number(number=enemy_l),
            right_enemy_princess_hp=number(number=enemy_r)))


def test_hand_and_next_card_use_the_simulator_names():
    obs, _allies = native_rl.live_observation(native_state(), elapsed=30.0, multiplier=1.0,
                                              towers=native_rl.TowerMemory())
    assert obs.hand == {"knight": 0, "archer": 1, "giant": 2, "arrows": 3}
    assert obs.next_card == "fireball"


def test_greyed_cards_stay_in_the_hand_and_only_the_mask_drops_them():
    """The simulator always encodes all four cards; dropping the unaffordable
    ones changed 16 decisions in 154 states. Affordability is the mask's job."""
    obs, _ = native_rl.live_observation(native_state(elixir=3, ready=(0, 3)), elapsed=30.0,
                                        multiplier=1.0, towers=native_rl.TowerMemory())
    assert obs.hand == {"knight": 0, "archer": 1, "giant": 2, "arrows": 3}
    mask = mask_from_live(obs, deck=STARTER)
    giant = slice(1 + 2 * TILES, 1 + 3 * TILES)
    assert not mask[giant].any()                             # 5 elixir, we have 3


def test_units_land_on_the_grid_the_bridge_expects():
    state = native_state(allies=[det("giant", 8, 3)], enemies=[det("hog_rider", 14, 20)])
    obs, allies = native_rl.live_observation(state, elapsed=30.0, multiplier=1.0,
                                             towers=native_rl.TowerMemory())
    # A unit standing in a tile sits at its centre, as the simulator places it.
    assert [(c.x, c.y, n) for c, n in allies] == [(8.5, 28.5, "giant")]
    assert [(t.cell.x, t.cell.y, t.name) for t in obs.tracks] == [(14.5, 11.5, "hog_rider")]


def bars(enemy_left=1.0, ally_left=1.0):
    return {"ally_left": ally_left, "ally_right": 1.0, "enemy_left": enemy_left,
            "enemy_right": 1.0}


def test_a_tower_bar_misread_as_empty_does_not_destroy_the_tower():
    towers = native_rl.TowerMemory()
    towers.update(bars(ally_left=0.9))
    for _ in range(20):                                     # a bar never found again
        seen = towers.update(bars(ally_left=0.0, enemy_left=0.0))
    assert seen["enemy_left"] == 1.0


def test_one_low_reading_does_not_stick():
    """Minions flying over the bar read 0.1 for a frame: not a tower at 10%."""
    towers = native_rl.TowerMemory()
    towers.update(bars(enemy_left=0.1))
    for _ in range(3):
        seen = towers.update(bars(enemy_left=1.0))
    assert seen["enemy_left"] == 1.0


def test_a_tower_taken_from_a_third_in_one_burst_is_destroyed():
    towers = native_rl.TowerMemory()
    for value in (1.0, 0.6, 0.3, 0.3, 0.3, 0.3):
        towers.update(bars(enemy_left=value))
    for _ in range(native_rl.TowerMemory.CONFIRM_FRAMES):
        seen = towers.update(bars(enemy_left=0.0))
    assert seen["enemy_left"] == 0.0


def test_a_tower_worn_down_then_gone_is_destroyed():
    towers = native_rl.TowerMemory()
    for value in (1.0, 0.6, 0.6, 0.6, 0.3, 0.3, 0.3, 0.12, 0.12, 0.12):
        towers.update(bars(enemy_left=value))
    for _ in range(native_rl.TowerMemory.CONFIRM_FRAMES):
        seen = towers.update(bars(enemy_left=0.0))
    assert seen["enemy_left"] == 0.0


def test_troops_are_kept_to_tiles_buildabot_can_deploy_on():
    """The simulator's mask allows row-0 corners and a deeper pocket than the
    game; a tap there is refused and retried every 2 s."""
    troop = native_rl.troop_tiles(enemy_down=())
    assert not troop[31 - 0, 0]                              # tile (0, 0): a corner
    assert troop[31 - 0, 8]                                  # tile (8, 0): behind the King
    assert troop[31 - 14, 3] and not troop[31 - 20, 3]       # our half only
    pocket = native_rl.troop_tiles(enemy_down=("left",))
    assert pocket[31 - 20, 3] and not pocket[31 - 20, 14]


def test_a_troop_sent_to_the_river_row_lands_one_row_back():
    state = native_state()
    action = 1 + 0 * TILES + 16 * 18 + 8                     # knight, grid (8, 16) = tile y 15
    plan = native_rl.plan_from_action(action, state, confidence=0.5)
    assert plan.tile == (8, 14)
    spell = native_rl.plan_from_action(1 + 3 * TILES + 16 * 18 + 8, state, confidence=0.5)
    assert spell.tile == (8, 15)                             # arrows land anywhere


def test_the_elixir_multiplier_follows_the_simulator():
    assert native_rl.multiplier_at(100) == 1.0
    assert native_rl.multiplier_at(200) == 2.0
    assert native_rl.multiplier_at(250) == 3.0


def test_an_action_becomes_a_hand_slot_and_a_buildabot_tile():
    state = native_state()
    action = 1 + 2 * TILES + 20 * 18 + 3                    # slot 2 (giant), grid (3, 20)
    plan = native_rl.plan_from_action(action, state, confidence=0.8)
    assert (plan.card, plan.slot, plan.tile) == ("giant", 2, (3, 11))


def test_the_simulator_refills_the_hand_at_the_end():
    """The convention the live hand order has to reproduce: in the simulator a
    played card leaves its slot and the next card joins on the right."""
    from sim.match import PlayerState
    player = PlayerState(side=1, deck=STARTER, queue=list(STARTER))
    player.draw()
    player.play(player.hand[1])
    assert player.hand == ["knight", "minions", "arrows", "fireball"]


def test_the_hand_keeps_the_simulator_order_when_the_game_refills_in_place():
    """The real game puts the next card in the played card's slot. Fed that
    order, the policy put 72% of its troops on tile (1, 13) in the simulator
    and lost 4 of 24 (0 of 24 in its own order) - the tic seen live."""
    order = native_rl.HandOrder()
    assert order.update(["knight", "archer", "giant", "arrows"]) == ["knight", "archer", "giant", "arrows"]
    assert order.update(["knight", "fireball", "giant", "arrows"]) == ["knight", "giant", "arrows", "fireball"]
    assert order.update(["minions", "fireball", "giant", "arrows"]) == ["giant", "arrows", "fireball", "minions"]
    order.reset()
    assert order.update(["giant", "knight", "arrows", "archer"]) == ["giant", "knight", "arrows", "archer"]


def test_live_observation_presents_the_hand_in_simulator_order():
    order = native_rl.HandOrder()
    towers = native_rl.TowerMemory()
    native_rl.live_observation(native_state(), 10.0, 1.0, towers, hand_order=order)
    refilled = native_state(hand=("knight", "fireball", "giant", "arrows"), nxt="archers")
    obs, _ = native_rl.live_observation(refilled, 14.0, 1.0, towers, hand_order=order)
    assert obs.hand == {"knight": 0, "giant": 1, "arrows": 2, "fireball": 3}


def test_an_action_on_a_simulator_slot_taps_that_card_in_the_game():
    state = native_state(hand=("knight", "fireball", "giant", "arrows"), nxt="archers")
    action = 1 + 3 * TILES + 20 * 18 + 3                    # simulator slot 3, grid (3, 20)
    plan = native_rl.plan_from_action(action, state, confidence=0.8,
                                      hand=["knight", "giant", "arrows", "fireball"])
    assert (plan.card, plan.slot, plan.tile) == ("fireball", 1, (3, 11))


def test_a_checkpoint_for_another_deck_is_refused():
    with pytest.raises(ValueError):
        native_rl.check_deck(["cannon", "fireball", "hog_rider", "ice_golem", "ice_spirit",
                              "musketeer", "skeletons", "the_log"],
                             ["knight", "archers", "minions", "arrows", "fireball", "giant",
                              "musketeer", "minipekka"])
    native_rl.check_deck(STARTER, ["knight", "archers", "minions", "arrows", "fireball",
                                   "giant", "musketeer", "minipekka"])

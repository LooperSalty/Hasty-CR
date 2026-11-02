"""ClashEnv with a deck other than Hog 2.6 for the learning seat.

The observation and action spaces only depend on there being eight cards, so
another deck fits the same network shapes - but only if every place that
assumed Hog 2.6 reads the seat's own deck instead: the hand one-hot, the
spell placement rule, and the card table.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.env import (ACTIONS, CARD_ACTIONS, DECK_26, NUM_SLOTS,  # noqa: E402
                     TILES, ClashEnv)

STARTER = ["knight", "archer", "minions", "arrows", "fireball", "giant",
           "musketeer", "minipekka"]
HAND_BASE = 3 + 2 * 2          # elixir, time, regen, then 2 x 2 lane fractions


def hand_one_hot(scalars):
    width = len(DECK_26)
    hand = scalars[HAND_BASE:HAND_BASE + NUM_SLOTS * width].reshape(NUM_SLOTS, width)
    nxt = scalars[HAND_BASE + NUM_SLOTS * width:HAND_BASE + (NUM_SLOTS + 1) * width]
    return hand, nxt


def test_default_env_still_plays_hog_26():
    env = ClashEnv(seed=1, opponent="simple")
    env.reset(seed=1)
    assert env.deck == DECK_26
    assert env.match.players[1].deck == DECK_26 or env.match.players[-1].deck == DECK_26


def test_custom_deck_is_played_and_encoded_against_its_own_order():
    env = ClashEnv(seed=1, opponent="simple", deck=STARTER)
    obs, info = env.reset(seed=2)
    seat = env.match.players[1]
    assert seat.deck == STARTER
    hand, nxt = hand_one_hot(obs["scalars"])
    for slot, card in enumerate(seat.hand[:NUM_SLOTS]):
        assert hand[slot, STARTER.index(card)] == 1.0
        assert hand[slot].sum() == 1.0
    assert nxt[STARTER.index(seat.next_card)] == 1.0


def test_arrows_and_fireball_may_target_the_enemy_side():
    env = ClashEnv(seed=1, opponent="simple", deck=STARTER)
    _, info = env.reset(seed=3)
    seat = env.match.players[1]
    seat.elixir = 10_000
    from sim.env import legal_mask
    mask = legal_mask(env.match, env._cards, 1)
    for slot, card in enumerate(seat.hand[:NUM_SLOTS]):
        offered = int(mask[1 + slot * TILES:1 + (slot + 1) * TILES].sum())
        if card in ("arrows", "fireball"):
            assert offered == TILES, f"{card} should reach every cell"
        else:
            assert 0 < offered < TILES, f"{card} is a troop, own half only"


def test_every_masked_action_is_accepted_with_the_starter_deck():
    env = ClashEnv(seed=1, opponent="simple", deck=STARTER)
    rng = np.random.default_rng(5)
    for seed in (21, 22):
        _, info = env.reset(seed=seed)
        for _ in range(120):
            legal = np.flatnonzero(info["action_mask"])
            action = int(rng.choice(legal))
            before = env.stats.illegal
            _, _, terminated, truncated, info = env.step(action)
            assert env.stats.illegal == before, ClashEnv.decode(action)
            if terminated or truncated:
                break


@pytest.mark.parametrize("bad", [
    STARTER[:7],
    STARTER[:7] + ["knight"],
    STARTER[:7] + ["not_a_card"],
])
def test_invalid_decks_are_refused(bad):
    with pytest.raises(ValueError):
        ClashEnv(seed=1, opponent="simple", deck=bad)


def test_action_space_is_unchanged():
    assert ACTIONS == 1 + 4 * TILES + (ACTIONS - CARD_ACTIONS)

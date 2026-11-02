"""Training against the decks the bot actually meets on ladder.

The `meta` pool is Path of Legends: top-ladder decks. Measured on 116 live
matches at 600-1100 trophies (2026-10-27), the losses cluster on cards that
pool barely holds - Lumberjack (65% of matches lost, 0 decks of 204),
Elite Barbarians (63%, 0 decks), Wizard (67%, 5 decks), P.E.K.K.A (78%,
9 decks). `live` is a scripted opponent like `meta`, drawing from decks
rebuilt from the units the runner logged in those matches, lost ones
weighted up.
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from build_live_decks import deck_from_units                     # noqa: E402
from sim.env import ClashEnv                                     # noqa: E402
from sim.gamedata import load_gamedata                           # noqa: E402
from sim.meta_decks import live_deck_pool                        # noqa: E402

CARDS = load_gamedata(11)
STARTER = ["knight", "archer", "minions", "arrows", "fireball", "giant", "musketeer", "minipekka"]


def test_units_become_simulator_cards_padded_with_spells():
    deck = deck_from_units(Counter({"lumberjack": 30, "elite_barbarian": 20, "skeleton": 9}), CARDS)
    assert len(deck) == 8 and len(set(deck)) == 8
    assert {"axe_man", "angry_barbarians", "skeletons"} <= set(deck)
    assert "fireball" in deck                                   # spells are never seen as units
    assert all(CARDS.get(card) is not None for card in deck)


def test_a_unit_seen_in_a_frame_or_two_is_a_misread_not_a_card():
    deck = deck_from_units(Counter({"pekka": 25, "wizard": 1}), CARDS)
    assert "pekka" in deck and "wizard" not in deck


def test_the_pool_keeps_repeats_as_weight_and_drops_unbuildable_decks(tmp_path):
    good = ["axe_man", "angry_barbarians", "skeletons", "fireball", "arrows", "zap", "knight", "archer"]
    path = tmp_path / "live_decks.json"
    path.write_text(json.dumps({"decks": [good, good, good[:7] + ["not_a_card"]]}), encoding="utf-8")
    pool = live_deck_pool(CARDS, path)
    assert [deck for _name, _style, deck in pool] == [good, good]


def test_an_opponent_spec_names_its_kind_and_card_strength():
    from sim.env import parse_opponent
    assert parse_opponent("meta") == ("meta", (1.0, 1.0))
    assert parse_opponent("live@1.33") == ("live", (1.33, 1.33))
    assert parse_opponent("live@1.0-1.4") == ("live", (1.0, 1.4))


def test_a_boost_makes_only_the_new_units_of_that_side_stronger():
    """Measured: the policy that wins 96% against `live` decks wins 52% when
    the opponent's units are 1.33x - two to three card levels, about the gap
    to the decks it meets at 1100 trophies - and that is its live record."""
    from types import SimpleNamespace
    from sim.opponents import boost_new_units

    def unit(side):
        return SimpleNamespace(side=side, hitpoints=100, max_hitpoints=100, damage=10)

    old, theirs, ours = unit(-1), unit(-1), unit(1)
    match = SimpleNamespace(battle=SimpleNamespace(entities={1: old, 2: theirs, 3: ours}))
    boost_new_units(match, side=-1, before={1}, factor=1.5)
    assert (theirs.hitpoints, theirs.max_hitpoints, theirs.damage) == (150, 150, 15)
    assert old.hitpoints == 100 and ours.hitpoints == 100


def test_the_live_opponent_draws_its_strength_per_episode(tmp_path, monkeypatch):
    deck = ["pekka", "wizard", "axe_man", "angry_barbarians", "fireball", "arrows", "zap", "skeletons"]
    path = tmp_path / "live_decks.json"
    path.write_text(json.dumps({"decks": [deck]}), encoding="utf-8")
    monkeypatch.setenv("HASTYCR_LIVE_DECKS", str(path))
    fixed = ClashEnv(seed=3, opponent="live@1.3", deck=STARTER)
    fixed.reset(seed=3)
    assert fixed.opponent.stat_boost == 1.3
    ranged = ClashEnv(seed=3, opponent="live", deck=STARTER, opponent_boost=(1.0, 1.4))
    boosts = set()
    for seed in range(6):
        ranged.reset(seed=seed)
        boosts.add(ranged.opponent.stat_boost)
    assert len(boosts) > 1 and all(1.0 <= b <= 1.4 for b in boosts)
    plain = ClashEnv(seed=3, opponent="live", deck=STARTER)
    plain.reset(seed=3)
    assert plain.opponent.stat_boost == 1.0


def test_the_live_opponent_plays_a_deck_from_its_pool(tmp_path, monkeypatch):
    deck = ["pekka", "wizard", "axe_man", "angry_barbarians", "fireball", "arrows", "zap", "skeletons"]
    path = tmp_path / "live_decks.json"
    path.write_text(json.dumps({"decks": [deck]}), encoding="utf-8")
    monkeypatch.setenv("HASTYCR_LIVE_DECKS", str(path))
    env = ClashEnv(seed=3, opponent="live", deck=STARTER)
    env.reset(seed=3)
    assert sorted(env.match.decks[1]) == sorted(deck)
    for _ in range(40):
        _obs, _reward, done, truncated, _info = env.step(0)
        assert not (done or truncated)
    env.close()

"""A simulator-trained policy as a strategy of the native runner, for any deck.

`brain.rl_policy` is the bridge cr_bot uses for Hog 2.6: the observation and
the legal-action mask, checked against the simulator cell for cell. This
module feeds it from what the native runner has instead - BuildABot's state
after `native_perception` (tracked units in BuildABot tiles, the hand, the
tower bars) - and turns the chosen action back into a hand slot and a tile,
as a `Plan` the runner already knows how to play.

Conventions, each pinned by tests/test_native_rl.py:

* a BuildABot tile (x, y) is grid cell (x, 31 - y), the grid the simulator
  acts on; both map to the same MuMu pixel;
* card names are BuildABot's except `archers`, which the simulator calls
  `archer`;
* a unit standing in tile (x, y) sits at its centre, (x + 0.5, y + 0.5) in
  the simulator's continuous frame;
* the hand is all four cards, greyed or not (the simulator always encodes
  four); affordability is the mask's job;
* the hand is numbered in the simulator's order - longest held first, the
  newest card on the right - not by the game's slots, which refill in place
  (HandOrder); the chosen card is then tapped wherever it sits in the game;
* troops are kept to tiles BuildABot can deploy on, and one sent to the
  river row lands one row back - the simulator allows both, the game not;
* tower bars sometimes read 0 when the bar is simply not found; a drop only
  counts after CONFIRM_FRAMES equal readings, and an empty bar only destroys
  a tower that was already below half (TowerMemory).

What it cannot see, honestly: unit hit points (live vision gives each unit
its card's full HP, as in cr_bot), and whatever perception misses.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

from beatdown_strategy import Plan
from brain.arena import to_grid
from brain.rl_policy import card_cost, mask_from_live, observation_from_live
from clashroyalebuildabot.constants import (ALLY_TILES, LEFT_PRINCESS_TILES,
                                            RIGHT_PRINCESS_TILES)
from sim.env import ACTIONS, CARD_ACTIONS, GRID_H, GRID_W, NUM_PLANES, NUM_SCALARS, TILES
from sim.match import DOUBLE_AT_MS, TRIPLE_AT_MS

# Card names the simulator spells differently from BuildABot.
BUILDABOT_TO_SIM = {"archers": "archer"}
SIM_TO_BUILDABOT = {sim: buildabot for buildabot, sim in BUILDABOT_TO_SIM.items()}
TOWER_BARS = {"ally_left": "left_ally_princess_hp", "ally_right": "right_ally_princess_hp",
              "enemy_left": "left_enemy_princess_hp", "enemy_right": "right_enemy_princess_hp"}
PHASE = "ia"
RIVER_ROW, LAST_ALLY_ROW = 15, 14          # BuildABot tile rows
# Cards that land anywhere, by name: only used to tell a spell from a troop
# when choosing whether a river-row tile needs moving back.
SPELL_NAMES = frozenset({"arrows", "fireball", "zap", "the_log", "rocket", "lightning",
                         "poison", "earthquake", "freeze", "tornado", "giant_snowball",
                         "barbarian_barrel", "goblin_barrel", "graveyard", "rage", "clone",
                         "mirror", "royal_delivery"})


def sim_name(card: str) -> str:
    return BUILDABOT_TO_SIM.get(card, card)


def buildabot_name(card: str) -> str:
    return SIM_TO_BUILDABOT.get(card, card)


def check_deck(trained_on, in_game) -> None:
    """The checkpoint must have been trained on the deck in play (any order)."""
    if sorted(trained_on) != sorted(sim_name(card) for card in in_game):
        raise ValueError(f"checkpoint trained on {sorted(trained_on)}, the deck in play is "
                         f"{sorted(sim_name(card) for card in in_game)}")


class TowerMemory:
    """Princess tower fractions that a misread bar cannot move.

    A drop counts once CONFIRM_FRAMES readings agree on it (Minions flying
    over a bar read it at 0.1 for a frame). An empty bar destroys the tower
    only if it was already below DESTROY_BELOW: a full bar read as empty is
    a bar that was not found. Towers never heal, so higher readings are
    ignored, and a destroyed tower stays destroyed.
    """

    CONFIRM_FRAMES = 3
    TOLERANCE = 0.05
    DESTROY_BELOW = 0.5

    def __init__(self):
        self.value = dict.fromkeys(TOWER_BARS, 1.0)
        self._pending: dict[str, tuple[float, int]] = {}

    def update(self, readings: dict) -> dict:
        for key, reading in readings.items():
            current = self.value.get(key, 1.0)
            if current <= 0.0 or reading >= current - self.TOLERANCE:
                self._pending.pop(key, None)
                continue
            seen, count = self._pending.get(key, (reading, 0))
            count = count + 1 if abs(seen - reading) <= self.TOLERANCE else 1
            self._pending[key] = (reading, count)
            if count < self.CONFIRM_FRAMES:
                continue
            self._pending.pop(key)
            if reading > 0.0:
                self.value[key] = reading
            elif current < self.DESTROY_BELOW:
                self.value[key] = 0.0
        return dict(self.value)


def troop_tiles(enemy_down) -> np.ndarray:
    """Grid cells (row, column) where BuildABot may deploy a troop.

    Its own half (row 0 only behind the King), the river row - moved one row
    back when played - and a fallen princess tower's pocket.
    """
    tiles = [tuple(tile) for tile in ALLY_TILES]
    tiles += [(x, RIVER_ROW) for x in range(GRID_W)]
    if "left" in enemy_down:
        tiles += [tuple(tile) for tile in LEFT_PRINCESS_TILES]
    if "right" in enemy_down:
        tiles += [tuple(tile) for tile in RIGHT_PRINCESS_TILES]
    grid = np.zeros((GRID_H, GRID_W), dtype=bool)
    for x, y in tiles:
        if 0 <= x < GRID_W and 0 <= y < GRID_H:
            grid[GRID_H - 1 - y, x] = True
    return grid


class HandOrder:
    """The hand in the order the simulator keeps it, from the game's slots.

    The game puts the next card in the slot the played card left; the
    simulator removes the played card and adds the next one on the right, so
    its slots run from the longest-held card to the newest. The policy reads
    its hand slot by slot and acts on slots, and fed the game's order it
    stopped playing as it had learned (72% of troops on one tile in the
    simulator, the tic seen live). Cards still held keep their place, new
    ones join on the right, in the game's slot order.
    """

    def __init__(self):
        self.cards: tuple = ()

    def reset(self) -> None:
        self.cards = ()

    def update(self, live_cards) -> list:
        live = [card for card in live_cards if card]
        kept = [card for card in self.cards if card in live]
        self.cards = tuple(kept + [card for card in live if card not in kept])
        return list(self.cards)


def live_observation(state, elapsed: float, multiplier: float, towers: TowerMemory,
                     hand_order: Optional[HandOrder] = None):
    """(obs, allies) in the shape `brain.rl_policy` reads, from BuildABot state.

    With `hand_order` the hand is numbered in the simulator's order rather
    than by the game's slots; `plan_from_action` then needs that order back.
    """
    cards = state.cards
    in_slots = [sim_name(cards[slot + 1].name) for slot in range(4)
                if cards[slot + 1].name != "blank"]
    ordered = hand_order.update(in_slots) if hand_order is not None else in_slots
    hand = {card: slot for slot, card in enumerate(ordered)}
    readings = {key: float(getattr(state.numbers, bar).number) for key, bar in TOWER_BARS.items()}
    seen = towers.update(readings)
    allies = [(_centre(d), d.unit.name) for d in state.allies]
    tracks = [SimpleNamespace(cell=_centre(d), name=d.unit.name) for d in state.enemies]
    obs = SimpleNamespace(
        allies=[cell for cell, _ in allies], tracks=tracks,
        elixir=float(state.numbers.elixir.number), elapsed=float(elapsed),
        multiplier=float(multiplier),
        ally_hp={"left": seen["ally_left"], "right": seen["ally_right"]},
        enemy_hp={"left": seen["enemy_left"], "right": seen["enemy_right"]},
        hand=hand, next_card=sim_name(cards[0].name))
    return obs, allies


def _centre(det):
    """The grid point of a unit standing in a BuildABot tile: the tile's centre."""
    return to_grid(det.position.tile_x + 0.5, det.position.tile_y - 0.5)


def plan_from_action(action: int, state, confidence: float,
                     hand: Optional[list] = None) -> Optional[Plan]:
    """A card action as the runner's Plan: hand slot, BuildABot tile, why.

    `hand` is the simulator-ordered hand the observation was numbered by
    (`HandOrder`); the Plan's slot is where that card sits in the game.
    """
    if not 1 <= action < CARD_ACTIONS:
        return None
    slot, cell = divmod(action - 1, TILES)
    grid_y, grid_x = divmod(cell, GRID_W)
    if hand is not None:
        if slot >= len(hand):
            return None
        names = [state.cards[index + 1].name for index in range(4)]
        card = buildabot_name(hand[slot])
        if card not in names:
            return None
        slot = names.index(card)
    card = state.cards[slot + 1].name
    tile_y = int(GRID_H - 1 - grid_y)
    if tile_y == RIVER_ROW and sim_name(card) not in SPELL_NAMES:
        tile_y = LAST_ALLY_ROW            # a troop cannot stand in the river
    tile = (int(grid_x), tile_y)
    return Plan(phase=PHASE, card=card, slot=slot, tile=tile,
                reason=f"IA entraînée : {card} en {tile} (confiance {confidence:.0%})")


def multiplier_at(elapsed: float) -> float:
    """The simulator's schedule: 2x from 2:00, 3x from 4:00 (in overtime)."""
    if elapsed * 1000 >= TRIPLE_AT_MS:
        return 3.0
    return 2.0 if elapsed * 1000 >= DOUBLE_AT_MS else 1.0


class RLStrategy:
    """`decide(state, now) -> Plan`, like BeatdownStrategy, from a checkpoint."""

    name = "ia"
    handles_spells = True            # the policy casts its own spells

    def __init__(self, checkpoint: Path, in_game_deck, device: Optional[str] = None,
                 min_confidence: float = 0.0):
        import torch
        from sim.train_ppo import build_network
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Tensors, numbers and the deck list only: no pickled code is loaded.
        blob = torch.load(Path(checkpoint), map_location=self.device, weights_only=True)
        self.deck = list(blob.get("deck") or [])
        check_deck(self.deck, in_game_deck)
        self.network = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS)
        self.network.load_state_dict(blob["state_dict"])
        self.network.to(self.device).eval()
        self.checkpoint = Path(checkpoint).name
        self.min_confidence = min_confidence
        self.started: Optional[float] = None
        self.towers = TowerMemory()
        self.hand_order = HandOrder()
        # Resolve the card table now: built lazily it took ~4 s inside the
        # first decision of the first match.
        card_cost(self.deck[0], self.deck)

    def start_match(self, now: float) -> None:
        self.started = now
        self.towers = TowerMemory()
        self.hand_order.reset()

    def played(self, card: str, tile, now: float) -> None:
        """Nothing to remember: the next observation shows the new hand."""

    @staticmethod
    def _deployable(mask: np.ndarray, obs) -> np.ndarray:
        """The simulator's mask, troops cut to tiles the live game accepts."""
        down = tuple(lane for lane, value in obs.enemy_hp.items() if value <= 0.0)
        troops = troop_tiles(down).reshape(-1)
        for card, slot in obs.hand.items():
            if card not in SPELL_NAMES:
                start = 1 + slot * TILES
                mask[start:start + TILES] &= troops
        return mask

    def decide(self, state, now: float) -> Plan:
        elapsed = 0.0 if self.started is None else max(0.0, now - self.started)
        obs, allies = live_observation(state, elapsed, multiplier_at(elapsed), self.towers,
                                       hand_order=self.hand_order)
        view = observation_from_live(obs, allies, deck=self.deck)
        mask = self._deployable(mask_from_live(obs, deck=self.deck), obs)
        if not mask[1:].any():
            return Plan(phase=PHASE, reason=f"IA : rien de jouable ({obs.elixir:.0f} élixir)")
        torch = self.torch
        with torch.no_grad():
            planes = torch.from_numpy(view["planes"]).unsqueeze(0).to(self.device)
            scalars = torch.from_numpy(view["scalars"]).unsqueeze(0).to(self.device)
            logits, _value = self.network(planes, scalars)
            logits = logits[0].float()
            logits[~torch.from_numpy(np.asarray(mask)).to(logits.device)] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.argmax(logits).item())
            confidence = float(probs[action].item())
        if action == 0 or confidence < self.min_confidence:
            return Plan(phase=PHASE, reason=f"IA : j'attends (confiance {confidence:.0%})")
        plan = plan_from_action(action, state, confidence, hand=list(self.hand_order.cards))
        return plan or Plan(phase=PHASE, reason="IA : capacité, pas de bouton en direct")

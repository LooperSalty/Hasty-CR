"""Let a live rules strategy (PekkaStrategy...) play the bottom seat of the simulator.

The strategies are written against BuildABot's view of the game: detections
with a unit name and a tile, the hand as cards with a cost, the elixir and
the princess tower bars. This builds that view from a `sim.match.Match` every
decision, asks the strategy, and plays its plan back into the match - so the
same rules that play on MuMu can be watched (and recorded) in sim.watch.

Tiles: BuildABot's x 0-17, y 0 at our back row; the simulator's tile y runs
the other way (0 at the top), so y_sim = 31 - y.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

from sim import arena as sim_arena
from sim.adapter import grid_to_point

# The simulator and BuildABot name these cards differently.
SIM_TO_BUILDABOT = {"archer": "archers"}
BUILDABOT_TO_SIM = {v: k for k, v in SIM_TO_BUILDABOT.items()}
PEKKA_DECK = ["knight", "pekka", "minions", "arrows", "inferno_tower", "archer", "musketeer",
              "minipekka"]


def _detection(entity) -> SimpleNamespace:
    x, y = sim_arena.to_tiles(entity.pos)
    target = ("buildings" if getattr(entity, "target_only_buildings", False)
              else "all" if getattr(entity, "attacks_air", False) else "ground")
    return SimpleNamespace(
        unit=SimpleNamespace(name=entity.name,
                             transport="air" if getattr(entity, "flying", False) else "ground",
                             target=target),
        position=SimpleNamespace(tile_x=int(x), tile_y=int(31 - y), conf=1.0))


def live_view(match, side: int = 1) -> SimpleNamespace:
    """The match as BuildABot's state for the bottom player."""
    player = match.players[side]

    def card(name):
        spec = match.cards.get(name)
        return SimpleNamespace(name=SIM_TO_BUILDABOT.get(name, name),
                               cost=getattr(spec, "cost", 0) if spec else 0)

    hand = list(player.hand[:4])
    allies, enemies = [], []
    for entity in match.battle.entities.values():
        if not getattr(entity, "alive", True) or getattr(entity, "is_tower", False):
            continue
        (allies if entity.side == side else enemies).append(_detection(entity))
    theirs = match.tower_fractions(-side)
    number = SimpleNamespace
    return SimpleNamespace(
        cards=[card(player.next_card or hand[0]), *[card(name) for name in hand]],
        ready=list(range(len(hand))), allies=allies, enemies=enemies,
        numbers=SimpleNamespace(
            elixir=number(number=player.elixir // 1000),
            left_enemy_princess_hp=number(number=theirs.get("left", 1.0)),
            right_enemy_princess_hp=number(number=theirs.get("right", 1.0))))


class RulesSeat:
    """`act(match)` like the simulator's opponents, driven by a live strategy."""

    def __init__(self, strategy, side: int = 1):
        self.strategy = strategy
        self.side = side
        self.last_reason = ""

    def reset(self) -> None:
        start = getattr(self.strategy, "start_match", None)
        if start:
            start(0.0)

    def act(self, match) -> Optional[tuple]:
        now = match.elapsed_ms / 1000.0
        plan = self.strategy.decide(live_view(match, self.side), now)
        self.last_reason = plan.reason
        if plan.card is None or plan.tile is None:
            return None
        card = BUILDABOT_TO_SIM.get(plan.card, plan.card)
        x, y = plan.tile
        point = grid_to_point(int(x), 31 - int(y), self.side)
        if not match.play_card(self.side, card, point):
            return None
        self.strategy.played(plan.card, plan.tile, now)
        return card, x, y, plan.reason

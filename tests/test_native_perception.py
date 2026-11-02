"""The native runner's board: detections -> tracks -> sides -> enemy elixir."""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "vendor" / "ClashRoyaleBuildABot")):
    if path not in sys.path:
        sys.path.insert(0, path)

from native_perception import Perception  # noqa: E402

DECK = [SimpleNamespace(name=n, units=[SimpleNamespace(name=u)]) for n, u in (
    ("knight", "knight"), ("archers", "archer"), ("minions", "minion"), ("giant", "giant"),
    ("musketeer", "musketeer"), ("minipekka", "minipekka"), ("arrows", None), ("fireball", None))]
for card in DECK:
    card.units = [u for u in card.units if u.name]


def seen(name, x, y, side=0, side_conf=0.6, conf=0.8):
    return SimpleNamespace(name=name, tile_x=x, tile_y=y, confidence=conf, side=side,
                           side_confidence=side_conf, box=(10, 20, 30, 40))


class FakeVision:
    def __init__(self):
        self.frames, self.error, self.last_spells = [], "", []

    def detect(self, frame):
        ours, theirs, spells = self.frames.pop(0)
        self.last_spells = spells
        return ours, theirs


def perception():
    vision = FakeVision()
    p = Perception(DECK, vision=vision, detector=None, upstream_units=None, log=lambda line: None)
    p.start_match(0.0)
    return p, vision


def battle_state():
    return SimpleNamespace(allies=[], enemies=[])


def test_a_troop_of_ours_misread_as_enemy_stays_ours():
    p, vision = perception()
    p.played("musketeer", 1.0)
    vision.frames = [([seen("musketeer", 4, 5)], [], []),
                     ([], [seen("musketeer", 4, 6, side=1, side_conf=0.9)], [])]
    state = battle_state()
    p.perceive(state, None, 1.2)
    p.perceive(state, None, 1.4)
    assert [d.unit.name for d in state.allies] == ["musketeer"]
    assert state.enemies == []


def test_a_new_enemy_troop_is_charged_once_and_its_box_kept():
    p, vision = perception()
    vision.frames = [([], [seen("giant", 9, 26, side=1)], []),
                     ([], [seen("giant", 9, 25, side=1)], [])]
    state = battle_state()
    p.perceive(state, None, 20.0)
    first = p.enemy.value
    p.perceive(state, None, 20.3)
    assert first == 5.0                                  # 10 - Giant
    assert p.enemy.value > first                         # same Giant: regenerating, not charged
    assert state.enemies[0].position.bbox == (10, 20, 30, 40)


def test_an_enemy_lost_then_seen_again_is_not_charged_twice():
    p, vision = perception()
    vision.frames = [([], [seen("giant", 9, 18, side=1)], []),
                     ([], [], []), ([], [], []),
                     ([], [seen("giant", 9, 15, side=1)], [])]
    state = battle_state()
    for now in (20.0, 20.4, 21.0, 22.0):        # hidden behind the bridge 1.6 s
        p.perceive(state, None, now)
    assert p.enemy.value > 5.0                  # 10 - one Giant, then regenerating


def test_our_spell_is_not_charged_to_them_but_theirs_is():
    p, vision = perception()
    p.played("fireball", 30.0)
    vision.frames = [([], [], [("fireball", 9, 24, 0.8, (1, 2, 3, 4))]),
                     ([], [], [("zap", 8, 8, 0.8, (1, 2, 3, 4))])]
    state = battle_state()
    p.perceive(state, None, 30.5)
    assert p.enemy.value == 10.0
    p.perceive(state, None, 31.0)
    assert p.enemy.value == 8.0                          # their Zap: -2


def test_tower_troops_are_on_neither_side():
    p, vision = perception()
    vision.frames = [([], [seen("dagger_duchess_tower", 9, 5, side=1)], [])]
    state = battle_state()
    p.perceive(state, None, 1.0)
    assert state.allies == [] and state.enemies == []


def test_units_spawned_by_an_enemy_troop_are_not_new_cards():
    """A Witch's skeletons, a Golem's golemites, an Elixir Golem's halves:
    billed as new cards they drained the estimate to 0 all match."""
    p, vision = perception()
    witch = seen("witch", 9, 22, side=1)
    vision.frames = [([], [witch], []),
                     ([], [witch, seen("skeleton", 8, 22, side=1),
                           seen("skeleton", 10, 22, side=1)], [])]
    state = battle_state()
    p.perceive(state, None, 50.0)
    after_witch = p.enemy.value                         # 10 - 5 (Witch)
    p.perceive(state, None, 50.5)
    assert p.enemy.value >= after_witch                 # skeletons: free


def test_a_dying_golem_does_not_bill_its_golemites():
    p, vision = perception()
    vision.frames = [([], [seen("golem", 4, 20, side=1)], []),
                     ([], [seen("golemite", 3.5, 19, side=1), seen("golemite", 4.5, 19, side=1)],
                      [])]
    state = battle_state()
    p.perceive(state, None, 51.0)
    assert p.enemy.value == 2.0                         # 10 - 8 (Golem)
    p.perceive(state, None, 51.3)
    assert p.enemy.value >= 2.0                         # golemites (billed 8 before): free


def test_goblins_out_of_a_goblin_barrel_are_not_billed_again():
    p, vision = perception()
    vision.frames = [([], [], [("goblin_barrel", 3, 6, 0.8, (1, 2, 3, 4))]),
                     ([], [seen("goblin", 3, 6, side=1), seen("goblin", 4, 6, side=1)], [])]
    state = battle_state()
    p.perceive(state, None, 40.0)
    after_barrel = p.enemy.value
    p.perceive(state, None, 41.0)
    assert after_barrel == 7.0                          # 10 - 3
    assert p.enemy.value >= after_barrel


def test_a_card_deployed_on_its_own_is_still_billed():
    p, vision = perception()
    vision.frames = [([], [seen("skeleton", 9, 25, side=1), seen("skeleton", 8, 25, side=1)], [])]
    state = battle_state()
    p.perceive(state, None, 40.0)
    assert p.enemy.value == 9.0                         # Skeletons: 1


def test_troops_deployed_next_to_a_tank_are_still_billed():
    p, vision = perception()
    giant = seen("giant", 9, 25, side=1)
    vision.frames = [([], [giant], []),
                     ([], [giant, seen("barbarian", 8, 26, side=1),
                           seen("barbarian", 10, 26, side=1)], [])]
    state = battle_state()
    p.perceive(state, None, 50.0)
    after_giant = p.enemy.value                         # 10 - 5
    p.perceive(state, None, 50.5)
    assert p.enemy.value < after_giant                  # Barbarians: 5, not free


def test_a_deployed_elixir_golem_is_billed():
    p, vision = perception()
    vision.frames = [([], [seen("elixir_golem", 9, 25, side=1)], [])]
    state = battle_state()
    p.perceive(state, None, 50.0)
    assert p.enemy.value == 7.0                         # 10 - 3


def upstream(name, x, y):
    return SimpleNamespace(unit=SimpleNamespace(name=name),
                           position=SimpleNamespace(tile_x=x, tile_y=y, conf=0.8,
                                                    bbox=(1, 2, 3, 4)))


def test_detections_in_the_scenery_outside_the_arena_are_dropped():
    """BuildABot's model reads the side decorations as a Knight at column -2
    or 19 now and then; the arena is columns 0-17."""
    p = Perception(DECK, vision=None, detector=None, upstream_units=None, log=lambda line: None)
    p.start_match(0.0)
    state = SimpleNamespace(allies=[upstream("knight", 19, 5), upstream("archer", 13, 8)],
                            enemies=[upstream("knight", -2, 13), upstream("bomber", 14, 16)])
    p.perceive(state, None, 1.0)
    assert [d.unit.name for d in state.allies] == ["archer"]
    assert [d.unit.name for d in state.enemies] == ["bomber"]


def test_a_shadow_of_our_unit_is_neither_ally_nor_enemy():
    p = Perception(DECK, vision=None, detector=None, upstream_units=None, log=lambda line: None)
    p.start_match(0.0)
    p.played("knight", 0.5, (7, 3))
    state = SimpleNamespace(allies=[upstream("knight", 7, 4)], enemies=[upstream("pekka", 8, 6)])
    p.perceive(state, None, 1.0)
    assert [d.unit.name for d in state.allies] == ["knight"]
    assert state.enemies == []


def test_a_unit_not_in_our_deck_behind_our_king_tower_is_scenery():
    """Live, a Lava Pup sat at (9,0) - our King Tower's back row - all match."""
    p = Perception(DECK, vision=None, detector=None, upstream_units=None, log=lambda line: None)
    p.start_match(0.0)
    state = SimpleNamespace(allies=[upstream("knight", 8, 0)],
                            enemies=[upstream("lava_pup", 9, 0), upstream("bomber", 14, 16)])
    p.perceive(state, None, 1.0)
    assert [d.unit.name for d in state.allies] == ["knight"]
    assert [d.unit.name for d in state.enemies] == ["bomber"]

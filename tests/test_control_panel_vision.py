"""The panel's 'what the bot sees' overlay: turning a detector State into boxes and text.

Detection itself is BuildABot's; what is tested here is the translation the
panel adds - coordinates from the detector's 368x652 frame back onto the
screenshot, and the summary a person reads - since a wrong scale would draw
every box in the wrong place and look like a detector fault.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from control_panel.vision import scale_bbox, seen_from_state, summary  # noqa: E402


def unit(name, conf, bbox):
    return SimpleNamespace(unit=SimpleNamespace(name=name),
                           position=SimpleNamespace(conf=conf, bbox=bbox))


def number(value, bbox=(0, 0, 10, 10)):
    return SimpleNamespace(number=value, bbox=bbox)


def state(**overrides):
    base = dict(
        allies=[unit("knight", 0.91, (100, 300, 120, 330))],
        enemies=[unit("giant", 0.64, (150, 100, 190, 150)), unit("archer", 0.42, (10, 10, 20, 20))],
        numbers=SimpleNamespace(
            elixir=number(7),
            left_enemy_princess_hp=number(0.72), right_enemy_princess_hp=number(0.0),
            left_ally_princess_hp=number(1.0), right_ally_princess_hp=number(0.5)),
        cards=[SimpleNamespace(name=n) for n in ("arrows", "knight", "giant", "minions", "fireball")],
        ready=[0, 3],
        screen=SimpleNamespace(name="in_game"),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_bbox_scales_from_detector_frame_to_screenshot():
    assert scale_bbox((0, 0, 368, 652), (1080, 1920)) == (0.0, 0.0, 1080.0, 1920.0)
    assert scale_bbox((184, 326, 368, 652), (540, 960)) == (270.0, 480.0, 540.0, 960.0)


def test_state_becomes_boxes_with_side_and_confidence():
    seen = seen_from_state(state(), (1080, 1920))
    labels = [(b.label, round(b.conf, 2), b.ally) for b in seen.boxes]
    assert labels == [("knight", 0.91, True), ("giant", 0.64, False), ("archer", 0.42, False)]
    assert seen.hand == ("knight", "giant", "minions", "fireball")
    assert seen.next_card == "arrows"
    assert seen.ready == (0, 3)
    assert seen.elixir == 7
    assert seen.towers["enemy_right"] == 0
    assert seen.screen == "in_game"


def test_summary_reads_like_a_sentence_with_percentages():
    text = summary(seen_from_state(state(), (1080, 1920)))
    assert "Élixir 7" in text
    assert "knight ✓" in text and "giant ✗" in text and "fireball ✓" in text
    assert "1 allié" in text and "2 ennemis" in text
    assert "giant 64 %" in text
    # Princess bars are fractions of full health, shown as percentages.
    assert "Tours ennemies 72 % / 0 %" in text
    assert "alliées 100 % / 50 %" in text


def test_outside_a_match_the_hand_is_not_reported():
    lobby = seen_from_state(state(screen=SimpleNamespace(name="lobby")), (1080, 1920))
    text = summary(lobby)
    assert "pas en match" in text
    assert "Main" not in text


def test_missing_state_is_described_not_crashed():
    seen = seen_from_state(None, (1080, 1920))
    assert seen.boxes == ()
    assert "aucune détection" in summary(seen).lower()

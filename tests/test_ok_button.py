"""The blue confirm button (OK / Continue) on the screens between matches.

Arena unlocks and King Tower level-ups end on a single blue OK button and
have no close cross, so popup_guard finds nothing and the runner used to sit
there until it relaunched the app. Fixtures are real 540x960 captures.
"""

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from ok_button import find_ok_button  # noqa: E402

SCREENS = ROOT / "tests" / "fixtures" / "screens"


def load(name: str, size=None) -> Image.Image:
    image = Image.open(SCREENS / name).convert("RGB")
    return image.resize(size) if size else image


def test_finds_the_ok_button_on_the_king_tower_level_up():
    spot = find_ok_button(load("arena_king_tower_ok.png"))
    assert spot is not None
    x, y = spot
    assert 220 <= x <= 320 and 770 <= y <= 815


def test_works_on_the_normalised_full_resolution_frame():
    spot = find_ok_button(load("arena_king_tower_ok.png", (1080, 1920)))
    assert spot is not None
    x, y = spot
    assert 440 <= x <= 640 and 1540 <= y <= 1630


def test_finds_the_ok_button_at_the_bottom_of_the_trophy_road():
    spot = find_ok_button(load("trophy_road_ok.png"))
    assert spot is not None
    x, y = spot
    assert 220 <= x <= 320 and 900 <= y <= 945


def test_no_false_button_in_the_lobby_or_in_battle():
    assert find_ok_button(load("lobby.png")) is None
    assert find_ok_button(load("battle.png")) is None
    # The arena loading screen fades to blue at the bottom; pressing anything
    # there could cancel the matchmaking.
    assert find_ok_button(load("matchmaking.png")) is None

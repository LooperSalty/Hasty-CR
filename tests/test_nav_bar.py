"""Getting back to the Battle tab from another main-menu tab.

Left on the Cards tab (after upgrading cards by hand), the runner saw an
unknown screen, found no popup to close and relaunched the game - which
reopened on the same tab, twice. The Battle tab button moves with the
selected tab, so it is found by its icon, not by a fixed point.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from PIL import Image  # noqa: E402

from nav_bar import find_battle_tab  # noqa: E402

SCREENS = ROOT / "tests" / "fixtures" / "screens"


def frame(name):
    return Image.open(SCREENS / name).convert("RGB").resize((1080, 1920))


def test_the_battle_tab_is_found_from_the_cards_tab():
    spot = find_battle_tab(frame("deck_tab.jpg"))
    assert spot is not None
    x, y = spot
    assert abs(x - 626) <= 12 and 1760 <= y <= 1860


def test_the_battle_tab_moves_with_the_selected_tab():
    shifted = Image.new("RGB", (1080, 1920))
    shifted.paste(frame("deck_tab.jpg").crop((0, 1700, 1080, 1920)).crop((80, 0, 1080, 220)),
                  (0, 1700))                                     # whole bar 80 px left
    x, _ = find_battle_tab(shifted)
    assert abs(x - 546) <= 12


def test_nothing_is_found_in_a_battle():
    assert find_battle_tab(frame("battle.png")) is None


def test_nothing_is_found_on_a_reward_screen():
    assert find_battle_tab(frame("trophy_road_ok.png")) is None

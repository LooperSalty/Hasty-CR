"""Victory or defeat, read from the end-of-game screen.

Both fixtures are real frames from MuMu: a defeat (-15 trophies, the
opponent's three gold crowns on top) and a victory (+30, our three gold
crowns above our banner). The layout is the same in both - opponent on
top, us below - so the banner order says nothing; the gold crowns do.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from PIL import Image  # noqa: E402

from match_result import banner_outcome  # noqa: E402

SCREENS = ROOT / "tests" / "fixtures" / "screens"
LOSS = SCREENS / "end_of_game_loss.png"
WIN = SCREENS / "end_of_game_win.png"


def test_a_real_defeat_reads_as_defeat():
    assert banner_outcome(Image.open(LOSS)) == "défaite"


def test_a_real_victory_reads_as_victory():
    """The frame an earlier banner-order rule called a defeat."""
    assert banner_outcome(Image.open(WIN)) == "victoire"


def test_a_frame_without_crowns_is_unknown():
    assert banner_outcome(Image.new("RGB", (1080, 1920), (90, 90, 90))) == "inconnu"


def test_any_frame_size_is_accepted():
    assert banner_outcome(Image.open(LOSS).resize((368, 652))) == "défaite"
    assert banner_outcome(Image.open(WIN).resize((540, 960))) == "victoire"


def test_an_unreadable_frame_is_unknown_not_an_error():
    assert banner_outcome(object()) == "inconnu"


def test_crowns_are_counted_per_side():
    from match_result import crowns
    assert crowns(Image.open(LOSS)) == (0, 3)
    assert crowns(Image.open(WIN)) == (3, 0)


def test_an_emote_bubble_hiding_a_crown_does_not_turn_a_defeat_into_a_tie():
    """1-2 defeat, the opponent's 'Bien joué !' bubble over one of their crowns."""
    from match_result import crowns
    frame = Image.open(SCREENS / "end_of_game_loss_emote.png")
    assert crowns(frame) == (1, 2)
    assert banner_outcome(frame) == "défaite"


def test_an_emote_picture_hiding_the_first_crown_still_counts_it():
    """1-2 defeat (2026-10-29), the opponent's King emote drawn over their
    left crown. Crowns fill the slots from the left, so a gold middle slot
    means the left one is won too."""
    from match_result import crowns
    frame = Image.open(SCREENS / "end_of_game_loss_emote_image.jpg")
    assert crowns(frame) == (1, 2)
    assert banner_outcome(frame) == "défaite"


def test_every_kept_live_frame_reads_as_its_checked_label():
    """Frames kept by the runner (35 live matches). Their labels were written
    by earlier readers; the four those got wrong were checked by eye and
    renamed, about ten more spot-checked."""
    import pytest
    folder = ROOT / "tmp" / "live" / "results"
    frames = sorted(folder.glob("*.jpg")) if folder.is_dir() else []
    labelled = [f for f in frames if f.stem.endswith(("_victoire", "_défaite"))]
    if not labelled:
        pytest.skip("no live result frames on this machine")
    wrong = [f.name for f in labelled
             if banner_outcome(Image.open(f)) != f.stem.rsplit("_", 1)[1]]
    assert wrong == []

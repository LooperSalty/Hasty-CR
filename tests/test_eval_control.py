"""Scoring detectors against units listed by eye, by name."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_control import name_scores  # noqa: E402


def test_names_are_matched_as_multisets():
    assert name_scores(["archer", "archer", "giant"], ["archer", "giant", "knight"]) == (2, 3, 3)


def test_an_empty_board_scores_nothing():
    assert name_scores([], []) == (0, 0, 0)

"""The overnight loop's bookkeeping: results from the runner log, run names."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import overnight_cycle as oc  # noqa: E402
from overnight_cycle import next_run_name, parse_results  # noqa: E402

LOG = """\
16:25:19 RESULT exit screen=end_of_game finished=34 outcome=victoire record=18V-16D
16:27:30 PLAY #712 knight tile=(3,13)
16:27:31 RESULT exit screen=end_of_game finished=35 outcome=défaite record=18V-17D
16:30:14 RESULT exit screen=lobby finished=36 outcome=inconnu record=18V-17D
16:33:41 STOP max-matches finished=36 actions=749
"""


def test_results_are_counted_from_the_runner_log():
    assert parse_results(LOG.splitlines()) == {"victoire": 1, "défaite": 1, "inconnu": 1}


def test_no_result_lines_count_nothing():
    assert parse_results(["16:00:00 PLAY #1 giant"]) == {"victoire": 0, "défaite": 0, "inconnu": 0}


def test_run_names_continue_the_series(tmp_path):
    (tmp_path / "starter_long3_best.pt").write_bytes(b"x")
    (tmp_path / "starter_long2_best.pt").write_bytes(b"x")
    (tmp_path / "starter_long_best.pt").write_bytes(b"x")
    assert next_run_name(tmp_path) == "starter_long4"


def test_the_first_run_name_without_any_previous(tmp_path):
    assert next_run_name(tmp_path) == "starter_long2"


def test_a_series_prefix_keeps_its_own_runs(tmp_path):
    """The weak-point series (live decks, stronger opponents) runs beside the
    old one: its names and its newest best must not mix with starter_long."""
    from overnight_cycle import newest_best
    (tmp_path / "starter_long7_best.pt").write_bytes(b"x")
    (tmp_path / "starter_weak1_best.pt").write_bytes(b"x")
    assert next_run_name(tmp_path, "starter_weak") == "starter_weak2"
    assert newest_best(tmp_path, "starter_weak").name == "starter_weak1_best.pt"
    assert newest_best(tmp_path).name == "starter_long7_best.pt"
    assert newest_best(tmp_path, "starter_none") is None


def test_the_climb_command_plays_the_chosen_strategy_and_deck():
    """2026-11-02: the P.E.K.K.A deck climbs with the hybrid (policy + rules)."""
    checkpoint = oc.ROOT / "tmp" / "rl" / "pekka_long2_best.pt"
    command = oc.climb_command(checkpoint, 12, strategy="hybride", deck="pekka")
    assert command[command.index("-Strategy") + 1] == "hybride"
    assert command[command.index("-Deck") + 1] == "pekka"
    assert command[command.index("-Checkpoint") + 1] == str(Path("tmp/rl/pekka_long2_best.pt"))
    assert command[command.index("-Matches") + 1] == "12"


def test_the_climb_command_defaults_to_the_policy_on_the_starter_deck():
    command = oc.climb_command(oc.ROOT / "tmp" / "rl" / "x_best.pt", 6)
    assert command[command.index("-Strategy") + 1] == "ia"
    assert command[command.index("-Deck") + 1] == "starter"


def test_the_train_command_passes_the_simulator_deck():
    deck = "knight,pekka,minions,arrows,inferno_tower,archer,musketeer,minipekka"
    command = oc.train_command(oc.ROOT / "tmp" / "rl" / "pekka_long2_best.pt", "pekka_long3",
                               3_000_000, sim_deck=deck)
    assert command[command.index("-Deck") + 1] == deck
    assert command[command.index("-Name") + 1] == "pekka_long3"
    assert "-Deck" not in oc.train_command(oc.ROOT / "tmp" / "rl" / "a.pt", "b", 1)

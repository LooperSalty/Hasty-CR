"""The control panel's pure logic: what it launches and what it reads back.

The window itself is thin; everything that decides *which* process starts with
*which* flags, or what a log line means, lives in plain functions so a wrong
flag fails here rather than as a bot that silently plays the wrong deck.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from control_panel.bots import (BOTS, DECKS, LaunchOptions,  # noqa: E402
                                bot_for_cmdline, bot_for_process, build_command,
                                is_match_end)
from control_panel.device_panel import instance_index  # noqa: E402
from control_panel.devicestate import (parse_device_online,  # noqa: E402
                                       parse_focused_package, parse_screen_size)
from control_panel.logstats import (LogStats, clean_line, is_debug,  # noqa: E402
                                    update_stats)

PYTHON = Path("C:/venv/python.exe")
ADB = Path("C:/mumu/adb.exe")


def options(**overrides) -> LaunchOptions:
    base = dict(bot="buildabot", deck="starter", matches=1, hours=1.0,
                no_queue=False, checkpoint=None, adb=ADB, serial="127.0.0.1:16384")
    base.update(overrides)
    return LaunchOptions(**base)


# --- launch commands -------------------------------------------------------

def test_buildabot_command_uses_native_runner_with_deck_and_limits():
    cmd = build_command(options(), PYTHON, ROOT)
    assert cmd[0] == str(PYTHON)
    assert cmd[1] == "-u"
    assert cmd[2].endswith("mumu_buildabot_native.py")
    assert cmd[cmd.index("--deck") + 1] == "starter"
    assert cmd[cmd.index("--adb") + 1] == str(ADB)
    assert cmd[cmd.index("--serial") + 1] == "127.0.0.1:16384"
    assert cmd[cmd.index("--max-matches") + 1] == "1"
    assert cmd[cmd.index("--hours") + 1] == "1.0"


def test_beatdown_bot_runs_the_native_runner_with_the_giant_strategy():
    cmd = build_command(options(bot="beatdown"), PYTHON, ROOT)
    assert cmd[2].endswith("mumu_buildabot_native.py")
    assert cmd[cmd.index("--strategy") + 1] == "beatdown"
    assert cmd[cmd.index("--deck") + 1] == "starter"
    assert "--strategy" not in build_command(options(bot="buildabot"), PYTHON, ROOT)


def test_zero_matches_means_unlimited():
    cmd = build_command(options(matches=0), PYTHON, ROOT)
    assert "--max-matches" not in cmd


def test_no_queue_is_passed_to_every_bot():
    for bot in ("buildabot", "rules"):
        assert "--no-queue" in build_command(options(bot=bot, no_queue=True), PYTHON, ROOT)
    assert "--no-queue" not in build_command(options(no_queue=False), PYTHON, ROOT)


def test_rules_bot_runs_cr_bot_without_checkpoint_or_deck(tmp_path):
    cmd = build_command(options(bot="rules"), PYTHON, tmp_path)
    assert cmd[2].endswith("cr_bot.py")
    assert "--rl" not in cmd
    assert "--deck" not in cmd


def test_live_bot_uses_the_trained_detector_once_it_exists(tmp_path):
    cmd = build_command(options(bot="rules"), PYTHON, tmp_path)
    assert cmd[cmd.index("--vision") + 1] == "buildabot"
    weights = tmp_path / "tmp" / "yolo" / "runs" / "cr_detector_s" / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"x")
    cmd = build_command(options(bot="rules"), PYTHON, tmp_path)
    assert cmd[cmd.index("--vision") + 1] == "yolo"


def test_rl_bot_requires_an_existing_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="checkpoint"):
        build_command(options(bot="rl"), PYTHON, ROOT)
    with pytest.raises(ValueError, match="checkpoint"):
        build_command(options(bot="rl", checkpoint=tmp_path / "missing.pt"), PYTHON, ROOT)
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"x")
    cmd = build_command(options(bot="rl", checkpoint=ckpt), PYTHON, ROOT)
    assert cmd[2].endswith("cr_bot.py")
    assert cmd[cmd.index("--rl") + 1] == str(ckpt)


@pytest.mark.parametrize("bad", [
    dict(bot="nope"),
    dict(deck="nope"),
    dict(matches=-1),
    dict(hours=0),
    dict(serial=""),
])
def test_invalid_options_are_refused(bad):
    with pytest.raises(ValueError):
        build_command(options(**bad), PYTHON, ROOT)


def test_every_bot_has_a_label_and_description():
    for spec in BOTS.values():
        assert spec.label and spec.description


def test_match_end_markers_per_bot():
    assert is_match_end("buildabot", "20:29:54 RESULT exit screen=bypass_end_of_game")
    assert is_match_end("rules", "2026-10-10 20:30:00 MATCH_END #1 dur=180s")
    assert is_match_end("rl", "2026-10-10 20:30:00 MATCH_END #2 dur=180s")
    assert not is_match_end("buildabot", "20:29:45 PLAY #10 archers")
    assert not is_match_end("rules", "20:29:54 RESULT exit screen=x")


def test_running_bots_are_recognised_from_their_command_line():
    assert bot_for_cmdline(["python.exe", "scripts/mumu_buildabot_native.py", "--deck", "starter"]) == "buildabot"
    assert bot_for_cmdline(["python.exe", r"C:\x\scripts\cr_bot.py", "--rl", "a.pt"]) == "rl"
    assert bot_for_cmdline(["python.exe", "scripts\\cr_bot.py", "--adb", "a"]) == "rules"
    assert bot_for_cmdline(["python.exe", "scripts/supervisor.py"]) == "rules"
    assert bot_for_cmdline(["python.exe", "-m", "sim.clone"]) is None
    assert bot_for_cmdline([]) is None


def test_only_scripts_of_this_checkout_count_as_bots(tmp_path):
    repo = tmp_path / "repo"
    native = str(repo / "scripts" / "mumu_buildabot_native.py")
    assert bot_for_process(["python.exe", "-u", native], tmp_path, repo) == "buildabot"
    # Relative paths resolve against the process's working directory.
    assert bot_for_process(["python.exe", "scripts/cr_bot.py"], repo, repo) == "rules"
    # A file with the same name elsewhere is not ours, and must not be killed.
    other = str(tmp_path / "copy" / "scripts" / "cr_bot.py")
    assert bot_for_process(["python.exe", other, "--rl", "a.pt"], tmp_path, repo) is None
    assert bot_for_process(["python.exe", "scripts/cr_bot.py"], tmp_path, repo) is None
    # Running pytest *on* a bot script is not a bot.
    assert bot_for_process(["python.exe", "-m", "pytest", "scripts/cr_bot.py"], repo, repo) is None


def test_panel_decks_match_the_native_runner_presets():
    import ast
    source = (ROOT / "scripts" / "mumu_buildabot_native.py").read_text(encoding="utf-8")
    presets = next(node for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.Assign)
                   and getattr(node.targets[0], "id", None) == "DECK_PRESETS")
    keys = {key.value for key in presets.value.keys}
    assert keys == set(DECKS)


def test_mumu_instance_index_from_adb_port():
    assert instance_index("127.0.0.1:16384") == 0
    assert instance_index("127.0.0.1:16416") == 1
    assert instance_index("127.0.0.1:16480") == 3
    assert instance_index("garbage") == 0


# --- log statistics ---------------------------------------------------------

def test_stats_count_plays_matches_and_errors_without_mutating():
    start = LogStats()
    lines = [
        "20:28:41 START native BuildABot scoring",
        "20:28:44 PLAY #1 arrows tile=(6,8)",
        "20:28:53 PLAY #2 archers tile=(6,0)",
        "20:29:54 RESULT exit screen=bypass_end_of_game",
        "20:30:03 QUEUE match=1",
        "2026-10-10 20:26:44 QUEUE #2",
        "2026-10-10 20:31:00 MATCH_END #1 dur=180s",
        "20:31:05 ERROR RuntimeError: boom",
    ]
    stats = start
    for line in lines:
        stats = update_stats(stats, line)
    assert start == LogStats()
    assert stats.plays == 2
    assert stats.queued == 2
    assert stats.finished == 2
    assert stats.errors == 1
    assert "boom" in stats.last_error
    assert stats.last_event.endswith("ERROR RuntimeError: boom")


def test_blank_lines_change_nothing():
    assert update_stats(LogStats(), "   ") == LogStats()


def test_ansi_colours_are_stripped_and_debug_lines_recognised():
    raw = ("\x1b[32m2026-10-10 20:33:40.403\x1b[0m | \x1b[34m\x1b[1mDEBUG   \x1b[0m | "
           "\x1b[36mclashroyalebuildabot.detectors.detector\x1b[0m - Setting state...")
    text = clean_line(raw)
    assert "\x1b" not in text
    assert text.startswith("2026-10-10 20:33:40.403 | DEBUG")
    assert is_debug(text)
    assert not is_debug("20:33:43 STOP max-matches matches=1 actions=27")


# --- device state -----------------------------------------------------------

def test_device_online_requires_the_serial_in_device_state():
    out = "List of devices attached\n127.0.0.1:16384\tdevice\nemulator-5554\toffline\n"
    assert parse_device_online(out, "127.0.0.1:16384")
    assert not parse_device_online(out, "emulator-5554")
    assert not parse_device_online(out, "127.0.0.1:16480")
    assert not parse_device_online("", "127.0.0.1:16384")


def test_focused_package_is_read_from_dumpsys():
    out = "  mCurrentFocus=Window{5b21a95 u0 com.supercell.clashroyale/com.supercell.titan.GameApp}\n"
    assert parse_focused_package(out) == "com.supercell.clashroyale"
    assert parse_focused_package("mCurrentFocus=null") is None
    assert parse_focused_package("") is None


def test_screen_size_prefers_the_override():
    assert parse_screen_size("Physical size: 1080x1920\n") == (1080, 1920)
    assert parse_screen_size("Physical size: 1080x1920\nOverride size: 540x960\n") == (540, 960)
    assert parse_screen_size("garbage") is None

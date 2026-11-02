"""The live view's buttons: stop after the match, stop now, start."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from control_panel import bot_control  # noqa: E402
from control_panel.processes import RunningBot  # noqa: E402


def test_stop_after_match_is_a_flag_the_runner_reads(tmp_path):
    folder = tmp_path / "live"
    assert not bot_control.stop_after_match_pending(folder)
    bot_control.request_stop_after_match(folder)
    assert (folder / "stop_after_match.flag").is_file()
    assert bot_control.stop_after_match_pending(folder)
    bot_control.cancel_stop_after_match(folder)
    assert not bot_control.stop_after_match_pending(folder)
    bot_control.cancel_stop_after_match(folder)          # cancelling twice is harmless


def test_the_flag_name_is_the_one_the_runner_watches():
    import mumu_buildabot_native as native
    assert bot_control.STOP_FLAG_NAME == native.STOP_FLAG_NAME


def test_stop_now_kills_only_native_runners_and_resets_the_display(tmp_path):
    bots = [RunningBot(11, "beatdown", 1.0), RunningBot(12, "rules", 2.0),
            RunningBot(13, "buildabot", 3.0)]
    killed, resets = [], []
    bot_control.request_stop_after_match(tmp_path)
    result = bot_control.stop_now(
        tmp_path, find=lambda: bots, kill=lambda bot: killed.append(bot.pid) or [],
        reset=lambda: resets.append(True) or "")
    assert killed == [11, 13]
    assert resets == [True]
    assert result.stopped == 2 and result.survivors == [] and result.display_error == ""
    assert not bot_control.stop_after_match_pending(tmp_path)   # nothing left to stop


def test_stop_now_reports_processes_it_could_not_kill(tmp_path):
    bots = [RunningBot(21, "beatdown", 1.0)]
    result = bot_control.stop_now(tmp_path, find=lambda: bots, kill=lambda bot: [21],
                                  reset=lambda: "adb absent")
    assert result.survivors == [21] and result.display_error == "adb absent"
    assert "21" in result.message() and "adb absent" in result.message()


def test_stop_now_with_nothing_running_says_so(tmp_path):
    result = bot_control.stop_now(tmp_path, find=lambda: [], kill=lambda bot: [],
                                  reset=lambda: "")
    assert result.stopped == 0
    assert "aucun" in result.message().lower()


def test_climb_command_runs_the_launcher_with_an_optional_match_cap():
    command = bot_control.climb_command(ROOT)
    assert command[0].lower().startswith("powershell")
    assert str(ROOT / "scripts" / "climb.ps1") in command
    assert "-Matches" not in command
    assert bot_control.climb_command(ROOT, 1)[-2:] == ["-Matches", "1"]


def test_start_refuses_when_a_runner_is_already_playing():
    started = []
    message = bot_control.start_climb(ROOT, find=lambda: [RunningBot(5, "beatdown", 1.0)],
                                      popen=lambda *a, **k: started.append(a))
    assert started == [] and "déjà" in message


def test_start_launches_the_climb_script():
    started = []
    message = bot_control.start_climb(ROOT, matches=1, find=lambda: [],
                                      popen=lambda command, **k: started.append(command))
    assert len(started) == 1 and "climb.ps1" in " ".join(started[0])
    assert "lancé" in message.lower()

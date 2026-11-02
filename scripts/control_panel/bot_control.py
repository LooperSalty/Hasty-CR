"""Start and stop the native runner from the live view's buttons.

* "Arrêter après la partie" drops a flag file the runner checks at each
  match's end and in the lobby; it deletes the flag itself when it stops.
* "Arrêter maintenant" kills every native runner of this checkout (the same
  scan the control panel uses, so a recycled pid is never hit) and resets the
  emulator's display, which a killed runner never gets to do.
* "Lancer" runs scripts/climb.ps1, detached, unless a runner is already up.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .processes import RunningBot, find_running_bots, kill_bot

STOP_FLAG_NAME = "stop_after_match.flag"   # mumu_buildabot_native.STOP_FLAG_NAME
NATIVE_BOTS = frozenset({"beatdown", "buildabot"})
DEFAULT_ADB = Path(r"C:\Program Files\Netease\MuMuPlayer\nx_device\15.0\shell\adb.exe")
DEFAULT_SERIAL = "127.0.0.1:16384"
ADB_TIMEOUT = 15
CREATE_NO_WINDOW = 0x08000000


@dataclass(frozen=True)
class StopResult:
    stopped: int
    survivors: list = field(default_factory=list)
    display_error: str = ""

    def message(self) -> str:
        if not self.stopped:
            text = "Aucun bot en cours."
        else:
            text = f"Bot arrêté ({self.stopped} processus)."
        if self.survivors:
            text += f" Impossible d'arrêter {self.survivors} (lancé en administrateur ?)."
        if self.display_error:
            text += f" Écran non réinitialisé : {self.display_error}"
        return text


# ------------------------------------------------------------ stop after match

def _flag(folder: Path) -> Path:
    return Path(folder) / STOP_FLAG_NAME


def stop_after_match_pending(folder: Path) -> bool:
    return _flag(folder).is_file()


def request_stop_after_match(folder: Path) -> None:
    flag = _flag(folder)
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("demandé depuis la vision du bot\n", encoding="utf-8")


def cancel_stop_after_match(folder: Path) -> None:
    _flag(folder).unlink(missing_ok=True)


# ------------------------------------------------------------------ stop now

def native_runners(find: Callable[[], list] = find_running_bots) -> list[RunningBot]:
    return [bot for bot in find() if bot.bot in NATIVE_BOTS]


def reset_display(adb: Optional[Path] = None, serial: Optional[str] = None) -> str:
    """`wm size reset` on the emulator; '' on success, else what went wrong."""
    adb = adb or Path(os.environ.get("HASTYCR_ADB", str(DEFAULT_ADB)))
    serial = serial or os.environ.get("HASTYCR_SERIAL", DEFAULT_SERIAL)
    try:
        result = subprocess.run([str(adb), "-s", serial, "shell", "wm", "size", "reset"],
                                capture_output=True, text=True, timeout=ADB_TIMEOUT,
                                creationflags=CREATE_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)
    return "" if result.returncode == 0 else (result.stderr or result.stdout).strip()


def stop_now(folder: Path, find: Callable[[], list] = find_running_bots,
             kill: Callable[[RunningBot], list] = kill_bot,
             reset: Callable[[], str] = reset_display) -> StopResult:
    """Kill every native runner now. Blocking (kill waits, adb too): run it off the UI thread."""
    runners = native_runners(find)
    survivors = [pid for bot in runners for pid in kill(bot)]
    # Nothing is left to stop after the match; a later start must not stop at once.
    cancel_stop_after_match(folder)
    display_error = reset() if runners else ""
    return StopResult(stopped=len(runners), survivors=survivors, display_error=display_error)


# --------------------------------------------------------------------- start

def climb_command(root: Path, matches: int = 0) -> list[str]:
    command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
               "-File", str(Path(root) / "scripts" / "climb.ps1")]
    if matches > 0:
        command += ["-Matches", str(matches)]
    return command


def start_climb(root: Path, matches: int = 0,
                find: Callable[[], list] = find_running_bots,
                popen: Callable = subprocess.Popen) -> str:
    if native_runners(find):
        return "Un bot joue déjà : arrête-le d'abord."
    try:
        popen(climb_command(root, matches), cwd=str(root), creationflags=CREATE_NO_WINDOW,
              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as error:
        return f"Lancement impossible : {error}"
    limit = f"{matches} partie(s)" if matches else "sans limite"
    return f"Bot lancé ({limit}), il apparaît ici dans quelques secondes."

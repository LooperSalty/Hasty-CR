"""What the emulator is doing, read over adb.

Parsing is kept apart from the adb calls so it can be tested on captured
output; `probe` does the calls, each with a timeout, and never raises - an
unreachable emulator is a state to show, not an error.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PACKAGE = "com.supercell.clashroyale"
# Hide the console window adb would otherwise flash on every poll.
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_FOCUS = re.compile(r"mCurrentFocus=Window\{\S+ \S+ ([\w.]+)/")
_SIZE = re.compile(r"(Physical|Override) size: (\d+)x(\d+)")


def parse_device_online(devices_output: str, serial: str) -> bool:
    for line in devices_output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == serial:
            return parts[1] == "device"
    return False


def parse_focused_package(dumpsys_output: str) -> Optional[str]:
    match = _FOCUS.search(dumpsys_output)
    return match.group(1) if match else None


def parse_screen_size(wm_output: str) -> Optional[tuple[int, int]]:
    sizes = {kind: (int(w), int(h)) for kind, w, h in _SIZE.findall(wm_output)}
    return sizes.get("Override") or sizes.get("Physical")


@dataclass(frozen=True)
class DeviceState:
    online: bool = False
    installed: bool = False
    foreground: Optional[str] = None
    size: Optional[tuple[int, int]] = None
    error: str = ""

    @property
    def game_in_front(self) -> bool:
        return self.foreground == PACKAGE


def _adb(adb: Path, *args: str, timeout: float = 8.0) -> str:
    result = subprocess.run([str(adb), *args], capture_output=True, text=True,
                            timeout=timeout, creationflags=NO_WINDOW,
                            encoding="utf-8", errors="replace")
    return result.stdout


def probe(adb: Path, serial: str) -> DeviceState:
    if not Path(adb).is_file():
        return DeviceState(error=f"adb introuvable : {adb}")
    try:
        _adb(adb, "connect", serial)
        if not parse_device_online(_adb(adb, "devices"), serial):
            return DeviceState(error=f"aucun émulateur sur {serial}")
        packages = _adb(adb, "-s", serial, "shell", "pm", "list", "packages", PACKAGE)
        focus = _adb(adb, "-s", serial, "shell", "dumpsys", "window")
        size = _adb(adb, "-s", serial, "shell", "wm", "size")
    except (OSError, subprocess.SubprocessError) as exc:
        return DeviceState(error=f"adb a échoué : {exc}")
    return DeviceState(online=True,
                       installed=f"package:{PACKAGE}" in packages,
                       foreground=parse_focused_package(focus),
                       size=parse_screen_size(size))


FAST_CAPTURE_SIZE = "540x960"


def set_screen_size(adb: Path, serial: str, size: Optional[str]) -> str:
    """`wm size <size>`, or `wm size reset` for None.

    A 1080x1920 screencap is 8.3MB and ~400ms; 540x960 is 2.1MB and ~120ms
    (measured in run.ps1). Both bots normalise frames and scale taps, so the
    smaller surface only makes their capture - and the emulator - lighter.
    `wm size` belongs to the device, so it must be reset when the bot stops.
    """
    try:
        return _adb(adb, "-s", serial, "shell", "wm", "size", size or "reset")
    except (OSError, subprocess.SubprocessError) as exc:
        return f"échec : {exc}"


def launch_game(adb: Path, serial: str) -> str:
    """Bring Clash Royale to the front; returns adb's output for the log.

    `monkey` resolves the launcher activity itself, so this keeps working when
    Supercell renames the activity class.
    """
    try:
        return _adb(adb, "-s", serial, "shell", "monkey", "-p", PACKAGE,
                    "-c", "android.intent.category.LAUNCHER", "1")
    except (OSError, subprocess.SubprocessError) as exc:
        return f"échec : {exc}"


def screenshot(adb: Path, serial: str) -> Optional[bytes]:
    try:
        result = subprocess.run([str(adb), "-s", serial, "exec-out", "screencap", "-p"],
                                capture_output=True, timeout=8.0, creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    data = result.stdout
    return data if data.startswith(b"\x89PNG") else None

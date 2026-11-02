"""Keep MuMu's window on the primary monitor.

MuMu reopens where it last was. On a second monitor with another display
scaling, PrintWindow (the bot's capture) returns a quarter of the game,
magnified: the runner saw an unknown screen and relaunched the game in a
loop all evening (2026-10-12). This moves the window back to the primary
monitor's top-left area.

    python scripts/mumu_window.py
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Optional

TARGET = (620, 0)                 # left/top on the primary monitor
SWP_NOSIZE, SWP_NOZORDER = 0x0001, 0x0004


def mumu_window(instance: str = "Android Device") -> Optional[int]:
    """The biggest visible top-level window titled with the instance or MuMu."""
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _param):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length:
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            if instance in buffer.value or "MuMu" in buffer.value:
                rect = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                area = (rect.right - rect.left) * (rect.bottom - rect.top)
                if area > 0:
                    found.append((area, hwnd))
        return True

    user32.EnumWindows(visit, 0)
    return max(found)[1] if found else None


def on_primary(hwnd: int) -> bool:
    user32 = ctypes.windll.user32
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return 0 <= rect.left < width and 0 <= rect.top < height


def to_primary(instance: str = "Android Device") -> str:
    hwnd = mumu_window(instance)
    if hwnd is None:
        return "no MuMu window"
    if on_primary(hwnd):
        return "already on the primary monitor"
    ctypes.windll.user32.SetWindowPos(hwnd, 0, TARGET[0], TARGET[1], 0, 0,
                                      SWP_NOSIZE | SWP_NOZORDER)
    return "moved to the primary monitor" if on_primary(hwnd) else "move failed"


if __name__ == "__main__":
    print(to_primary(sys.argv[1] if len(sys.argv) > 1 else "Android Device"))

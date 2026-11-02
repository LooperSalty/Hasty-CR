import ctypes
import sys
import threading
import time

import numpy as np
import pytest

from scripts.studio import capture
from scripts.studio.capture import ThreadedSurfaceGrabber


@pytest.mark.skipif(sys.platform != "win32", reason="Windows display scaling")
def test_capture_reads_windows_in_physical_pixels():
    """On a monitor scaled to 150%, a DPI-unaware process gets the window's
    size divided by 1.5 while PrintWindow still draws it full size: the grab
    held the top-left two thirds of the game, magnified (2026-10-12, and
    measured again 2026-10-27 with MuMu on the 4K screen)."""
    capture.ensure_dpi_awareness()
    user32 = ctypes.windll.user32
    user32.GetThreadDpiAwarenessContext.restype = ctypes.c_void_p
    user32.GetAwarenessFromDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    awareness = user32.GetAwarenessFromDpiAwarenessContext(user32.GetThreadDpiAwarenessContext())
    assert awareness == 2                                  # per-monitor


def test_finding_and_grabbing_a_surface_both_turn_dpi_awareness_on(monkeypatch):
    calls = []
    monkeypatch.setattr(capture, "ensure_dpi_awareness", lambda: calls.append(1))
    monkeypatch.setattr(capture, "_top_level", lambda: [])
    list(capture.find_surfaces())
    assert calls == [1]
    monkeypatch.setattr(capture.SurfaceGrabber, "_allocate", lambda self, w, h: None)
    capture.SurfaceGrabber(capture.Surface(0, 10, 10, "test"))
    assert calls == [1, 1]


def test_slow_capture_does_not_block_reader_or_mutate_published_frame():
    blocked = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    class Capture:
        frames = 0
        stale_seconds = 0

        def __init__(self, surface):
            self.buffer = np.zeros((2, 2, 4), dtype=np.uint8)

        def grab(self):
            self.frames += 1
            if self.frames == 2:
                blocked.set()
                release.wait(3)
            self.buffer.fill(self.frames)
            return self.buffer

        def close(self):
            closed.set()

    worker = ThreadedSurfaceGrabber(None, factory=Capture)
    try:
        assert blocked.wait(3)
        started = time.monotonic()
        first = worker.grab()
        assert time.monotonic() - started < .1
        assert np.all(first == 1)
        release.set()
        deadline = time.monotonic() + 3
        while worker.frames < 2 and time.monotonic() < deadline:
            time.sleep(.005)
        assert worker.frames >= 2
        assert np.all(first == 1)
    finally:
        release.set()
        worker.close()
    assert closed.is_set()

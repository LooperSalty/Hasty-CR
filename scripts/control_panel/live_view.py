"""'Vision du bot': a HUD drawn over MuMu, and its decisions beside it.

Two windows read the JSON the native runner publishes after each frame it
decided on (tmp/live/native_view.json):

* the HUD - a transparent, click-through layer laid exactly over MuMu's
  render surface, boxing every unit the bot read with its confidence, ringing
  where it last played, and showing both elixir bars;
* the panel - docked beside MuMu, with the phase of its plan, the sentence
  saying why, its hand and its last decisions, and the buttons that start
  the bot, stop it after the match or at once, and show or hide the HUD.

The bot publishes ~10 times a second; the HUD redraws at 60 Hz and moves
each box along the course it was last seen on (live_view_model.BoxFollower),
so boxes glide with their troop instead of jumping from read to read.

Nothing here talks to the emulator and the bot's own capture (PrintWindow on
MuMu's surface) never sees this layer, so watching costs it nothing.

    python -m control_panel.live_view      (from scripts/), or vision.cmd
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PyQt6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel, QProgressBar,
                             QPushButton, QWidget)

from . import bot_control
from .live_view_model import BoxFollower, ViewReader, dock_geometry, read_view
from .widgets import STYLE

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "tmp" / "live"
INSTANCE = os.environ.get("HASTYCR_INSTANCE", "Android Device")
PANEL_WIDTH = 380
FRAME_W, FRAME_H = 1080, 1920       # the bot's coordinates
PHASE_COLOURS = {"lancement": "#e3b341", "push": "#f0883e", "défense": "#f85149",
                 "économie": "#58a6ff", "cycle": "#8b949e"}
ALLY, ENEMY, TARGET = QColor("#58a6ff"), QColor("#f85149"), QColor("#e3b341")
PANEL_MS, HUD_MS, SURFACE_MS = 250, 16, 500
# The bot publishes every ~0.1 s while it plays: older than this, it is not
# playing any more (stopped, frozen) and its boxes would be lies.
HUD_STALE_SECONDS = 1.5
START_SETTLE_MS = 5000


def mumu_window_rect() -> Optional[tuple[int, int, int, int]]:
    """The biggest visible top-level window whose title names the MuMu instance."""
    if sys.platform != "win32":
        return None
    user32 = ctypes.windll.user32
    found: list[tuple[int, tuple[int, int, int, int]]] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _param):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length:
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            if INSTANCE in buffer.value or "MuMu" in buffer.value:
                rect = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                area = (rect.right - rect.left) * (rect.bottom - rect.top)
                if area > 0:
                    found.append((area, (rect.left, rect.top, rect.right, rect.bottom)))
        return True

    user32.EnumWindows(visit, 0)
    return max(found)[1] if found else None


def surface_rect() -> Optional[tuple[int, int, int, int]]:
    """MuMu's Android render surface (left, top, width, height), physical pixels."""
    try:
        from studio.capture import _rect, find_surface
        return _rect(find_surface(instance=INSTANCE).hwnd)
    except Exception:
        return None


def to_logical(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Win32 rects are physical pixels; Qt places windows in logical ones."""
    screen = QGuiApplication.primaryScreen()
    ratio = screen.devicePixelRatio() if screen else 1.0
    return tuple(round(value / ratio) for value in rect)


class Hud(QWidget):
    """Transparent, click-through layer over MuMu's render surface."""

    def __init__(self):
        super().__init__(None, Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
                         | Qt.WindowType.WindowTransparentForInput
                         | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.data: Optional[dict] = None
        self.enabled = True
        self.follower = BoxFollower()
        self.font_label = QFont("Segoe UI", 8)
        self.font_label.setBold(True)
        self.font_badge = QFont("Segoe UI", 11)
        self.font_badge.setBold(True)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self.hide()

    def place(self, rect: Optional[tuple[int, int, int, int]]) -> None:
        if not self.enabled or rect is None or rect[2] <= 0 or rect[3] <= 0:
            self.hide()
            return
        x, y, w, h = to_logical(rect)
        if self.geometry().getRect() != (x, y, w, h):
            self.setGeometry(x, y, w, h)
        if not self.isVisible():
            self.show()

    def set_data(self, data: Optional[dict], changed: bool, now: float) -> None:
        if changed and data is not None:
            self.follower.update(data, received=now)
        self.data = data
        if self.isVisible():
            self.update()

    def paintEvent(self, event) -> None:      # noqa: N802 - Qt naming
        data = self.data
        if not data or data.get("screen") != "in_game":
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        sx, sy = self.width() / FRAME_W, self.height() / FRAME_H
        painter.setFont(self.font_label)
        for box in self.follower.frame(time.time()):
            self._unit(painter, ALLY if box.side == "ally" else ENEMY, box, sx, sy)
        target = data.get("target")
        if target:
            painter.setPen(QPen(TARGET, 3))
            painter.drawEllipse(QPointF(target[0] * sx, target[1] * sy), 16, 16)
        self._badges(painter, data, sx, sy)
        painter.end()

    @staticmethod
    def _unit(painter: QPainter, colour: QColor, box, sx: float, sy: float) -> None:
        x0, y0, x1, y1 = box.rect
        rect = QRectF(x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy)
        painter.setPen(QPen(colour, 2))
        painter.drawRect(rect)
        text = f"{box.name} {box.conf * 100:.0f}%"
        width = painter.fontMetrics().horizontalAdvance(text) + 8
        chip = QRectF(rect.left(), rect.top() - 15, width, 14)
        fill = QColor(colour)
        fill.setAlpha(200)
        painter.fillRect(chip, fill)
        painter.setPen(QColor("white"))
        painter.drawText(chip, Qt.AlignmentFlag.AlignCenter, text)

    def _badges(self, painter: QPainter, data: dict, sx: float, sy: float) -> None:
        painter.setFont(self.font_badge)
        enemy = data.get("enemy_elixir")
        if enemy is not None:
            self._badge(painter, QRectF(8, 150 * sy, 150, 26), ENEMY,
                        f"Adversaire ≈ {enemy:.1f} élixir")
        self._badge(painter, QRectF(8, 1610 * sy, 150, 26), QColor("#a371f7"),
                    f"Moi : {data.get('elixir', 0)} élixir")

    @staticmethod
    def _badge(painter: QPainter, rect: QRectF, colour: QColor, text: str) -> None:
        fill = QColor(colour)
        fill.setAlpha(210)
        painter.fillRect(rect, fill)
        painter.setPen(QColor("white"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


class DecisionPanel(QWidget):
    """Phase, reasoning, both elixir bars, hand, recent decisions and the controls."""

    message = pyqtSignal(str)          # from worker threads: shown under the buttons

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Décisions du bot")
        self.start_button = QPushButton("▶  Lancer le bot")
        self.after_button = QPushButton("⏸  Arrêter après la partie")
        self.after_button.setCheckable(True)
        self.stop_button = QPushButton("⏹  Arrêter maintenant")
        self.hud_button = QPushButton("HUD affiché")
        self.hud_button.setCheckable(True)
        self.hud_button.setChecked(True)
        self.notice = QLabel("")
        self.notice.setWordWrap(True)
        self.notice.setObjectName("hint")
        self.message.connect(self._done)
        self.phase = QLabel("—")
        self.phase.setStyleSheet("font-size:15pt; font-weight:bold;")
        self.reason = QLabel("—")
        self.reason.setWordWrap(True)
        self.elixir = self._bar("#a371f7", "Mon élixir %v / 10")
        self.enemy_elixir = self._bar("#f85149", "Adversaire ≈ %v / 10")
        self.hand = [QLabel("—") for _ in range(4)]
        hand_row = QHBoxLayout()
        for label in self.hand:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hand_row.addWidget(label)
        self.next_card = QLabel("Suivante : —")
        self.towers = QLabel("Tours : —")
        self.seen = QLabel("—")
        self.seen.setWordWrap(True)
        self.recent = QLabel("—")
        self.recent.setWordWrap(True)
        self.recent.setObjectName("hint")
        self.status = QLabel("—")
        self.status.setObjectName("hint")
        grid = QGridLayout(self)
        grid.addWidget(self.start_button, 0, 0)
        grid.addWidget(self.hud_button, 0, 1)
        grid.addWidget(self.after_button, 1, 0, 1, 2)
        grid.addWidget(self.stop_button, 2, 0, 1, 2)
        grid.addWidget(self.notice, 3, 0, 1, 2)
        for row, widget in enumerate((self.phase, self.reason, self.elixir, self.enemy_elixir),
                                     start=4):
            grid.addWidget(widget, row, 0, 1, 2)
        grid.addLayout(hand_row, 8, 0, 1, 2)
        grid.addWidget(self.next_card, 9, 0)
        grid.addWidget(self.towers, 9, 1)
        grid.addWidget(QLabel("<b>Ce qu'il voit</b>"), 10, 0, 1, 2)
        grid.addWidget(self.seen, 11, 0, 1, 2)
        grid.addWidget(QLabel("<b>Dernières décisions</b>"), 12, 0, 1, 2)
        grid.addWidget(self.recent, 13, 0, 1, 2)
        grid.setRowStretch(14, 1)
        grid.addWidget(self.status, 15, 0, 1, 2)

    def set_controls_enabled(self, enabled: bool) -> None:
        self.start_button.setEnabled(enabled)
        self.stop_button.setEnabled(enabled)

    def _done(self, text: str) -> None:
        """A background control finished (queued here from its thread)."""
        self.notice.setText(text)
        # A launched runner needs a few seconds to appear in the process list:
        # keep Start greyed that long, or a second click would not see it.
        QTimer.singleShot(START_SETTLE_MS, lambda: self.set_controls_enabled(True))

    def show_stop_pending(self, pending: bool) -> None:
        self.after_button.setChecked(pending)
        self.after_button.setText("⏸  Arrêt prévu après la partie (cliquer pour annuler)"
                                  if pending else "⏸  Arrêter après la partie")

    def show_hud_state(self, shown: bool) -> None:
        self.hud_button.setChecked(shown)
        self.hud_button.setText("HUD affiché" if shown else "HUD masqué")

    @staticmethod
    def _bar(colour: str, label: str) -> QProgressBar:
        bar = QProgressBar()
        bar.setRange(0, 10)
        bar.setFormat(label)
        bar.setStyleSheet("QProgressBar{border:1px solid #30363d;border-radius:5px;"
                          "text-align:center;height:20px;}"
                          f"QProgressBar::chunk{{background:{colour};border-radius:4px;}}")
        return bar

    def show_view(self, view) -> None:
        if view is None:
            self.status.setText("Aucune donnée : lance le bot (scripts\\climb.ps1).")
            return
        data = view.data
        record = data.get("record") or {}
        tally = f"{record.get('victoire', 0)} V / {record.get('défaite', 0)} D"
        self.status.setText(f"Bot arrêté (dernière donnée il y a {view.age:.0f} s) · {tally}"
                            if view.stale
                            else f"En direct · {data.get('strategy', '?')} · "
                                 f"{data.get('finished', 0)} match(s) ({tally}), "
                                 f"{data.get('actions', 0)} cartes")
        phase = data.get("phase") or "—"
        self.phase.setText(phase.capitalize())
        self.phase.setStyleSheet(f"font-size:15pt; font-weight:bold; "
                                 f"color:{PHASE_COLOURS.get(phase, '#e6edf3')};")
        self.reason.setText(data.get("reason", "—"))
        self.elixir.setValue(int(data.get("elixir", 0)))
        enemy = data.get("enemy_elixir")
        self.enemy_elixir.setValue(round(enemy) if enemy is not None else 0)
        self.enemy_elixir.setFormat(f"Adversaire ≈ {enemy:.1f} / 10" if enemy is not None
                                    else "Adversaire : hors match")
        for label, card in zip(self.hand, data.get("hand", [])):
            colour = "#3fb950" if card["ready"] else "#6e7681"
            label.setText(f"<span style='color:{colour}'>{card['name']}<br>"
                          f"{'jouable' if card['ready'] else 'pas prête'}</span>")
        self.next_card.setText(f"Suivante : {data.get('next_card', '—')}")
        towers = data.get("towers", {})
        self.towers.setText("Tours adv. {:.0%} / {:.0%}".format(
            towers.get("enemy_left", 0), towers.get("enemy_right", 0)) if towers else "Tours : —")
        self.seen.setText(self._seen_text(data))
        self.recent.setText("<br>".join(data.get("recent", [])[:8]) or "—")

    @staticmethod
    def _seen_text(data: dict) -> str:
        def items(units, colour):
            return [f"<span style='color:{colour}'>{u['name']} {u['conf'] * 100:.0f}%</span>"
                    for u in units]
        parts = items(data.get("enemies", []), "#f85149") + items(data.get("allies", []), "#58a6ff")
        return " · ".join(parts) or "Aucune unité en vue"


class LiveView:
    """Owns both windows and the timers that feed them."""

    def __init__(self, folder: Path = LIVE):
        self.folder = folder
        self.reader = ViewReader(folder)
        self.settings = QSettings("HastyCR", "live_view")
        self.hud = Hud()
        self.panel = DecisionPanel()
        self._wire_buttons()
        self._timers = []
        for interval, slot in ((HUD_MS, self.refresh_hud), (PANEL_MS, self.refresh_panel),
                               (SURFACE_MS, self.follow_mumu)):
            timer = QTimer()
            timer.timeout.connect(slot)
            timer.start(interval)
            self._timers.append(timer)
        self.follow_mumu()
        self.dock_panel()
        self.panel.show()

    def _wire_buttons(self) -> None:
        panel = self.panel
        panel.start_button.clicked.connect(
            lambda: self._background(lambda: bot_control.start_climb(ROOT), "Lancement…"))
        panel.stop_button.clicked.connect(
            lambda: self._background(lambda: bot_control.stop_now(self.folder).message(),
                                     "Arrêt du bot…"))
        panel.after_button.clicked.connect(self.toggle_stop_after_match)
        panel.hud_button.clicked.connect(self.toggle_hud)
        shown = self.settings.value("hud", True, type=bool)
        self.hud.set_enabled(shown)
        panel.show_hud_state(shown)
        panel.show_stop_pending(bot_control.stop_after_match_pending(self.folder))

    def _background(self, work, busy: str) -> None:
        """Run a blocking control (process scan, kill, adb) off the UI thread.

        Start and stop stay disabled until it is done: a double click must not
        launch two bots, nor a stop race a start that has not shown up yet.
        """
        panel = self.panel
        panel.notice.setText(busy)
        panel.set_controls_enabled(False)

        def run():
            try:
                text = work()
            except Exception as error:              # shown, never raised into Qt
                text = f"Erreur : {error}"
            panel.message.emit(text)

        threading.Thread(target=run, daemon=True).start()

    def toggle_stop_after_match(self) -> None:
        if bot_control.stop_after_match_pending(self.folder):
            bot_control.cancel_stop_after_match(self.folder)
            self.panel.notice.setText("Arrêt après la partie annulé.")
        else:
            bot_control.request_stop_after_match(self.folder)
            self.panel.notice.setText("Le bot s'arrêtera à la fin de la partie en cours "
                                      "(ou tout de suite s'il est au menu).")
        self.panel.show_stop_pending(bot_control.stop_after_match_pending(self.folder))

    def toggle_hud(self) -> None:
        shown = not self.hud.enabled
        self.hud.set_enabled(shown)
        self.settings.setValue("hud", shown)
        self.panel.show_hud_state(shown)
        if shown:
            self.follow_mumu()

    def follow_mumu(self) -> None:
        self.hud.place(surface_rect() if self.hud.enabled else None)

    def dock_panel(self) -> None:
        screen = QApplication.primaryScreen().availableGeometry()
        area = (screen.left(), screen.top(), screen.right(), screen.bottom())
        mumu = mumu_window_rect()
        x, y, w, h = dock_geometry(to_logical(mumu) if mumu else None, area, PANEL_WIDTH)
        self.panel.setGeometry(x, y, w, h)

    def refresh_hud(self) -> None:
        if not self.hud.enabled:
            return
        now = time.time()
        poll = self.reader.poll(now)
        view = poll.view
        live = view is not None and view.age <= HUD_STALE_SECONDS
        self.hud.set_data(view.data if live else None, poll.changed, now)

    def refresh_panel(self) -> None:
        self.panel.show_view(read_view(self.folder, now=time.time()))
        # The runner deletes the flag when it acts on it: follow that.
        self.panel.show_stop_pending(bot_control.stop_after_match_pending(self.folder))


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    app.setQuitOnLastWindowClosed(True)
    view = LiveView()
    app.view = view                            # keep a reference for the app's lifetime
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

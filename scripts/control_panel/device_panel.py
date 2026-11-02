"""The emulator box: where adb is, which instance, and what it is showing."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QVBoxLayout)

from . import devicestate
from .devicestate import DeviceState
from .widgets import GREEN, GREY, ORANGE, RED, Background, dot

DEFAULT_ADB = Path(r"C:\Program Files\Netease\MuMuPlayer\nx_device\15.0\shell\adb.exe")
# The first instance of a fresh MuMu install. Both bots run emulator.verify()
# before any tap, so a wrong instance is refused rather than played on.
DEFAULT_SERIAL = "127.0.0.1:16384"
FIRST_MUMU_PORT, MUMU_PORT_STEP = 16384, 32
PROBE_MS = 5000
# Five adb calls per probe, dumpsys among them: back off while a bot plays.
PROBE_BUSY_MS = 15000


def mumu_manager(adb: Path) -> Optional[Path]:
    for folder in Path(adb).parents:
        candidate = folder / "nx_main" / "MuMuManager.exe"
        if candidate.is_file():
            return candidate
    return None


def instance_index(serial: str) -> int:
    """MuMu numbers instances by adb port: 16384 is 0, then one every 32."""
    try:
        port = int(serial.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return 0
    return max(0, (port - FIRST_MUMU_PORT) // MUMU_PORT_STEP)


class DevicePanel(QGroupBox):
    log = pyqtSignal(str)
    state_changed = pyqtSignal(object)

    def __init__(self, settings, parent=None):
        super().__init__("Émulateur", parent)
        self._settings = settings
        self.state = DeviceState()
        self._probe = Background(self)
        self._probe.done.connect(self._on_probe)
        self._action = Background(self)
        self._action.done.connect(self._on_action)

        self.adb_edit = QLineEdit(settings.value("adb", os.environ.get("HASTYCR_ADB", str(DEFAULT_ADB))))
        self.serial_edit = QLineEdit(settings.value("serial", os.environ.get("HASTYCR_SERIAL", DEFAULT_SERIAL)))
        self.link_label = QLabel(dot(GREY, "pas encore vérifié"))
        self.game_label = QLabel(dot(GREY, "—"))
        self.size_label = QLabel("—")

        form = QFormLayout()
        form.addRow("adb :", self.adb_edit)
        form.addRow("Instance :", self.serial_edit)
        form.addRow("Connexion :", self.link_label)
        form.addRow("Clash Royale :", self.game_label)
        form.addRow("Écran :", self.size_label)

        refresh = QPushButton("Actualiser")
        refresh.clicked.connect(self.refresh)
        start_mumu = QPushButton("Démarrer MuMu")
        start_mumu.clicked.connect(self._start_mumu)
        open_game = QPushButton("Ouvrir Clash Royale")
        open_game.clicked.connect(self._open_game)
        buttons = QHBoxLayout()
        for button in (refresh, start_mumu, open_game):
            buttons.addWidget(button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(PROBE_MS)
        QTimer.singleShot(200, self.refresh)

    def set_busy(self, busy: bool) -> None:
        interval = PROBE_BUSY_MS if busy else PROBE_MS
        if self._timer.interval() != interval:
            self._timer.setInterval(interval)

    def adb(self) -> Path:
        return Path(self.adb_edit.text().strip().strip('"'))

    def serial(self) -> str:
        return self.serial_edit.text().strip()

    def save(self) -> None:
        self._settings.setValue("adb", str(self.adb()))
        self._settings.setValue("serial", self.serial())

    def refresh(self) -> None:
        self._probe.submit(devicestate.probe, self.adb(), self.serial())

    def _on_probe(self, state) -> None:
        if isinstance(state, Exception):
            state = DeviceState(error=str(state))
        self.state = state
        if not state.online:
            self.link_label.setText(dot(RED, state.error or "hors ligne"))
            self.game_label.setText(dot(GREY, "—"))
            self.size_label.setText("—")
        else:
            self.link_label.setText(dot(GREEN, f"connecté ({self.serial()})"))
            if not state.installed:
                self.game_label.setText(dot(RED, "non installé"))
            elif state.game_in_front:
                self.game_label.setText(dot(GREEN, "ouvert, au premier plan"))
            else:
                self.game_label.setText(dot(ORANGE, "installé, pas au premier plan"))
            self.size_label.setText(f"{state.size[0]}×{state.size[1]}" if state.size else "?")
        self.state_changed.emit(state)

    def _start_mumu(self) -> None:
        manager = mumu_manager(self.adb())
        if manager is None:
            self.log.emit("[panneau] MuMuManager.exe introuvable à côté de adb")
            return
        index = str(instance_index(self.serial()))
        self.log.emit(f"[panneau] démarrage de l'instance MuMu {index}…")
        self._action.submit(_run, [str(manager), "control", "-v", index, "launch"])

    def _open_game(self) -> None:
        self.log.emit("[panneau] ouverture de Clash Royale…")
        self._action.submit(devicestate.launch_game, self.adb(), self.serial())

    def _on_action(self, result) -> None:
        text = str(result).strip()
        if text:
            self.log.emit(f"[panneau] {text.splitlines()[-1]}")
        QTimer.singleShot(1500, self.refresh)


def _run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=60,
                            creationflags=devicestate.NO_WINDOW,
                            encoding="utf-8", errors="replace")
    return (result.stdout or result.stderr or "").strip()

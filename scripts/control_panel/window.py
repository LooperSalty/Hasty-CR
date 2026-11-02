"""The control panel window: choose a bot, start it, watch it, stop it."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
                             QPushButton, QSpinBox, QSplitter, QVBoxLayout, QWidget)

from . import devicestate
from .bots import BOTS, DECKS, LaunchOptions, build_command, is_match_end
from .device_panel import DevicePanel
from .logstats import LogStats, clean_line, is_debug, update_stats
from .processes import BotRunner, default_env, find_running_bots, kill_bot
from .vision import BotVision, summary
from .widgets import GREEN, GREY, ORANGE, Background, ScreenPreview, dot

SCAN_MS, CLOCK_MS, PREVIEW_MS = 2000, 1000, 1200
# While a bot plays, its own screen capture is most of its decision loop, so
# the preview backs off rather than compete with it for adb.
PREVIEW_BUSY_SECONDS = 5.0
MAX_LOG_LINES = 3000


def latest_checkpoint(root: Path) -> str:
    """Newest .pt under checkpoints/ only.

    Never tmp/ (scratch that training overwrites, see run.ps1) nor runs/
    (ultralytics writes YOLO weights there, which are not a policy).
    """
    found = [p for p in (root / "checkpoints").rglob("*.pt") if p.is_file()]
    return str(max(found, key=lambda p: p.stat().st_mtime)) if found else ""


def grab_image(adb: Path, serial: str):
    """Screenshot decoded off the UI thread; QImage, unlike QPixmap, allows that."""
    from PyQt6.QtGui import QImage
    data = devicestate.screenshot(adb, serial)
    image = QImage.fromData(data, "PNG") if data else QImage()
    return None if image.isNull() else image


def grab_and_analyse(vision: BotVision, adb: Path, serial: str, deck_key: str):
    """(annotated QImage, Seen), or None if no screenshot could be taken."""
    data = devicestate.screenshot(adb, serial)
    return vision.analyse(data, deck_key) if data else None


def venv_python(root: Path) -> Path:
    candidate = root / ".venvs" / "buildabot" / "Scripts" / "python.exe"
    return candidate if candidate.is_file() else Path(sys.executable)


class MainWindow(QMainWindow):
    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.settings = QSettings("HastyCR", "ControlPanel")
        self.runner = BotRunner(self)
        self.stats = LogStats()
        self.current_bot: Optional[str] = None
        self.started_at: Optional[float] = None
        self.stop_after_match = False
        self.external: list = []
        self.last_preview = 0.0
        self.resized = False          # True while the panel holds the device at 540x960
        self.vision = BotVision()

        self.setWindowTitle("Hasty-CR — Panneau de contrôle des bots")
        self.resize(1180, 760)
        self.device = DevicePanel(self.settings)
        self.device.log.connect(self.append_log)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._left_column())
        splitter.addWidget(self._right_column())
        splitter.setSizes([430, 750])
        self.setCentralWidget(splitter)

        self.runner.line.connect(self._on_line)
        self.runner.started.connect(self._on_started)
        self.runner.finished.connect(self._on_finished)

        self._scan = Background(self)
        self._scan.done.connect(self._on_scan)
        self._shot = Background(self)
        self._shot.done.connect(self._on_shot)
        self._killer = Background(self)
        self._killer.done.connect(self._on_killed)
        self._scan.submit(find_running_bots)     # don't wait 2s to learn of one
        for interval, slot in ((SCAN_MS, lambda: self._scan.submit(find_running_bots)),
                               (CLOCK_MS, self._refresh_status),
                               (PREVIEW_MS, self._request_preview)):
            timer = QTimer(self)
            timer.timeout.connect(slot)
            timer.start(interval)
        self._on_bot_changed()
        self._refresh_status()

    # --- layout -------------------------------------------------------------

    def _left_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.addWidget(self.device)
        layout.addWidget(self._bot_box())
        layout.addWidget(self._status_box())
        layout.addStretch(1)
        return column

    def _bot_box(self) -> QGroupBox:
        box = QGroupBox("Bot")
        self.bot_combo = QComboBox()
        for spec in BOTS.values():
            self.bot_combo.addItem(spec.label, spec.key)
        self._select(self.bot_combo, self.settings.value("bot", "buildabot"))
        self.bot_combo.currentIndexChanged.connect(self._on_bot_changed)
        self.bot_hint = QLabel()
        self.bot_hint.setObjectName("hint")
        self.bot_hint.setWordWrap(True)

        self.deck_combo = QComboBox()
        for key, text in DECKS.items():
            self.deck_combo.addItem(text.split(" (")[0], key)
            self.deck_combo.setItemData(self.deck_combo.count() - 1, text,
                                        Qt.ItemDataRole.ToolTipRole)
        self._select(self.deck_combo, self.settings.value("deck", "starter"))

        self.ckpt_edit = QLineEdit(self.settings.value("checkpoint", "") or latest_checkpoint(self.root))
        browse = QPushButton("…")
        browse.setFixedWidth(32)
        browse.clicked.connect(self._browse_checkpoint)
        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(self.ckpt_edit)
        ckpt_row.addWidget(browse)

        self.matches_spin = QSpinBox()
        self.matches_spin.setRange(0, 999)
        self.matches_spin.setSpecialValueText("illimité")
        self.matches_spin.setValue(self._number("matches", 1, int))
        self.hours_spin = QDoubleSpinBox()
        self.hours_spin.setRange(0.1, 24.0)
        self.hours_spin.setSingleStep(0.5)
        self.hours_spin.setSuffix(" h max")
        self.hours_spin.setValue(self._number("hours", 1.0, float))
        self.noqueue_check = QCheckBox("Ne pas lancer de partie (je lance un amical)")
        self.noqueue_check.setChecked(self.settings.value("no_queue", "false") == "true")
        self.fast_check = QCheckBox("Capture rapide (540×960 pendant que le bot joue)")
        self.fast_check.setToolTip("Allège l'émulateur et accélère le bot ; l'image est un peu "
                                   "moins nette. La résolution est rétablie à l'arrêt du bot.")
        self.fast_check.setChecked(self.settings.value("fast_capture", "true") == "true")

        form = QFormLayout()
        form.addRow("Bot :", self.bot_combo)
        form.addRow(self.bot_hint)          # spans both columns so it wraps fully
        form.addRow("Deck :", self.deck_combo)
        form.addRow("Checkpoint :", ckpt_row)
        form.addRow("Matchs :", self.matches_spin)
        form.addRow("Durée :", self.hours_spin)
        form.addRow("", self.noqueue_check)
        form.addRow("", self.fast_check)

        self.start_button = QPushButton("▶  Démarrer")
        self.start_button.setObjectName("start")
        self.start_button.clicked.connect(self.start_bot)
        self.after_button = QPushButton("Arrêter après ce match")
        self.after_button.setToolTip("Le bot finit le match en cours puis s'arrête.")
        self.after_button.clicked.connect(self._arm_stop_after_match)
        self.stop_button = QPushButton("■  Arrêter maintenant")
        self.stop_button.setObjectName("stop")
        self.stop_button.setToolTip("Arrête tous les bots, y compris ceux lancés hors du panneau.")
        self.stop_button.clicked.connect(self.stop_all)
        buttons = QGridLayout()
        buttons.addWidget(self.start_button, 0, 0, 1, 2)
        buttons.addWidget(self.after_button, 1, 0)
        buttons.addWidget(self.stop_button, 1, 1)

        layout = QVBoxLayout(box)
        layout.addLayout(form)
        layout.addLayout(buttons)
        return box

    def _status_box(self) -> QGroupBox:
        box = QGroupBox("État")
        self.state_label = QLabel()
        self.uptime_label = QLabel("—")
        self.matches_label = QLabel("—")
        self.plays_label = QLabel("—")
        self.errors_label = QLabel("—")
        self.event_label = QLabel("—")
        self.event_label.setWordWrap(True)
        self.event_label.setObjectName("hint")
        form = QFormLayout(box)
        form.addRow("Bot :", self.state_label)
        form.addRow("Durée :", self.uptime_label)
        form.addRow("Matchs :", self.matches_label)
        form.addRow("Cartes jouées :", self.plays_label)
        form.addRow("Erreurs :", self.errors_label)
        form.addRow("Dernier évènement :", self.event_label)
        return box

    def _right_column(self) -> QWidget:
        column = QSplitter(Qt.Orientation.Horizontal)

        preview_box = QGroupBox("Écran du jeu")
        self.preview = ScreenPreview()
        self.preview_check = QCheckBox("Aperçu en direct")
        self.preview_check.setChecked(self.settings.value("preview", "true") == "true")
        self.preview_check.toggled.connect(self._on_preview_toggled)
        self.vision_check = QCheckBox("Vision du bot (unités, % de confiance, main, élixir)")
        self.vision_check.setToolTip(
            "Passe chaque capture dans le détecteur que le bot utilise, avec le deck choisi, "
            "et dessine ce qu'il reconnaît. Bleu = allié, rouge = ennemi, vert = carte jouable.")
        self.vision_check.setChecked(self.settings.value("vision", "true") == "true")
        self.vision_label = QLabel("—")
        self.vision_label.setObjectName("hint")
        self.vision_label.setWordWrap(True)
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.addWidget(self.preview_check)
        preview_layout.addWidget(self.vision_check)
        preview_layout.addWidget(self.preview, 1)
        preview_layout.addWidget(self.vision_label)

        log_box = QGroupBox("Journal")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(MAX_LOG_LINES)
        self.debug_check = QCheckBox("Afficher les détails (DEBUG)")
        clear = QPushButton("Effacer")
        clear.clicked.connect(self.log_view.clear)
        bar = QHBoxLayout()
        bar.addWidget(self.debug_check)
        bar.addStretch(1)
        bar.addWidget(clear)
        log_layout = QVBoxLayout(log_box)
        log_layout.addLayout(bar)
        log_layout.addWidget(self.log_view, 1)

        column.addWidget(preview_box)
        column.addWidget(log_box)
        column.setSizes([300, 450])
        return column

    def _number(self, key: str, default, cast):
        """A saved number, or `default` if the stored value is unusable."""
        try:
            return cast(self.settings.value(key, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _select(combo: QComboBox, key) -> None:
        index = combo.findData(key)
        if index >= 0:
            combo.setCurrentIndex(index)

    # --- actions ------------------------------------------------------------

    def _options(self) -> LaunchOptions:
        text = self.ckpt_edit.text().strip().strip('"')
        checkpoint = Path(text) if text else None
        if checkpoint is not None and not checkpoint.is_absolute():
            checkpoint = self.root / checkpoint   # the bot runs from the repo root
        return LaunchOptions(
            bot=self.bot_combo.currentData(), deck=self.deck_combo.currentData(),
            matches=self.matches_spin.value(), hours=self.hours_spin.value(),
            no_queue=self.noqueue_check.isChecked(), checkpoint=checkpoint,
            adb=self.device.adb(), serial=self.device.serial())

    def start_bot(self) -> None:
        # Re-scan now rather than trust the last 2s poll: a bot started from a
        # terminal a moment ago must still block a second one.
        own = self.runner.pid
        self.external = [b for b in find_running_bots() if b.pid != own]
        if self.runner.running or self.external:
            QMessageBox.information(self, "Déjà en cours",
                                    "Un bot tourne déjà. Arrête-le avant d'en lancer un autre.")
            self._refresh_status()
            return
        options = self._options()
        try:
            command = build_command(options, venv_python(self.root), self.root)
        except ValueError as exc:
            QMessageBox.warning(self, "Impossible de démarrer", str(exc))
            return
        if self.device.state.online and not self.device.state.installed:
            QMessageBox.critical(self, "Mauvaise instance",
                                 f"Clash Royale n'est pas installé sur {options.serial}. "
                                 "Vérifie l'instance dans le cadre Émulateur.")
            return
        if not self.device.state.online:
            answer = QMessageBox.question(
                self, "Émulateur injoignable",
                "L'émulateur ne répond pas. Démarrer le bot quand même ?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._save_settings()
        self.stats = LogStats()
        self.stop_after_match = False
        self.current_bot = options.bot
        self.append_log(f"[panneau] démarrage : {BOTS[options.bot].label}")
        if self.fast_check.isChecked() and self.device.state.online:
            devicestate.set_screen_size(options.adb, options.serial, devicestate.FAST_CAPTURE_SIZE)
            self.resized = True
            self.append_log("[panneau] écran en 540×960 pendant le match (capture rapide)")
        try:
            self.runner.start(command, self.root, default_env(options.serial))
        except RuntimeError as exc:
            QMessageBox.warning(self, "Impossible de démarrer", str(exc))
        self._refresh_status()

    def _arm_stop_after_match(self) -> None:
        if not self.runner.running:
            self.append_log("[panneau] aucun bot lancé depuis le panneau")
            return
        self.stop_after_match = True
        self.append_log("[panneau] arrêt prévu à la fin du match en cours")
        self._refresh_status()

    def stop_all(self) -> None:
        self.stop_after_match = False
        if self.runner.running:
            self.append_log("[panneau] arrêt du bot…")
            self.runner.stop()
        targets = list(self.external)
        if targets:
            for bot in targets:
                self.append_log(f"[panneau] arrêt du bot externe (PID {bot.pid})…")
            self._killer.submit(lambda: {bot.pid: kill_bot(bot, self.root) for bot in targets})

    def _on_killed(self, result) -> None:
        if isinstance(result, Exception):
            self.append_log(f"[panneau] échec de l'arrêt : {result}")
        else:
            for pid, alive in result.items():
                if alive:
                    self.append_log(f"[panneau] PID {pid} : processus {alive} toujours actifs "
                                    "(lancés en administrateur ?)")
        self._scan.submit(find_running_bots)

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choisir un checkpoint", str(self.root),
                                              "Checkpoints PyTorch (*.pt)")
        if path:
            self.ckpt_edit.setText(path)

    def _on_bot_changed(self) -> None:
        spec = BOTS[self.bot_combo.currentData()]
        self.bot_hint.setText(spec.description)
        self.deck_combo.setEnabled(spec.uses_deck)
        self.ckpt_edit.setEnabled(spec.needs_checkpoint)

    def _save_settings(self) -> None:
        self.device.save()
        values = {"bot": self.bot_combo.currentData(), "deck": self.deck_combo.currentData(),
                  "matches": self.matches_spin.value(),
                  "hours": self.hours_spin.value(),
                  "no_queue": "true" if self.noqueue_check.isChecked() else "false",
                  "fast_capture": "true" if self.fast_check.isChecked() else "false",
                  "vision": "true" if self.vision_check.isChecked() else "false",
                  "preview": "true" if self.preview_check.isChecked() else "false"}
        # An empty checkpoint is not remembered, so the newest one under
        # checkpoints/ is picked up again once training produces it.
        if self.ckpt_edit.text().strip():
            values["checkpoint"] = self.ckpt_edit.text().strip()
        for key, value in values.items():
            self.settings.setValue(key, value)

    # --- bot output ---------------------------------------------------------

    def append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def _on_line(self, raw: str) -> None:
        line = clean_line(raw)
        self.stats = update_stats(self.stats, line)
        if self.debug_check.isChecked() or not is_debug(line):
            self.append_log(line)
        if self.stop_after_match and is_match_end(self.current_bot, line):
            self.append_log("[panneau] match terminé : arrêt du bot")
            self.stop_after_match = False
            self.runner.stop()

    def _on_started(self, pid: int) -> None:
        self.started_at = time.monotonic()
        self.append_log(f"[panneau] bot lancé (PID {pid})")
        self._refresh_status()

    def _on_finished(self, code: int) -> None:
        self.append_log(f"[panneau] bot arrêté (code {code})")
        self.started_at = None
        self.stop_after_match = False
        self._restore_screen()
        self._refresh_status()

    def _restore_screen(self) -> None:
        if self.resized:
            self.resized = False
            devicestate.set_screen_size(self.device.adb(), self.device.serial(), None)
            self.append_log("[panneau] résolution de l'écran rétablie (1080×1920)")

    def _on_scan(self, result) -> None:
        if isinstance(result, Exception):
            return
        own = self.runner.pid
        self.external = [bot for bot in result if bot.pid != own]

    # --- status and preview ---------------------------------------------------

    def _refresh_status(self) -> None:
        running = self.runner.running
        if running:
            text = "arrêt prévu après ce match" if self.stop_after_match else "en cours"
            colour = ORANGE if self.stop_after_match else GREEN
            self.state_label.setText(dot(colour, f"{BOTS[self.current_bot].label} — {text}"))
        elif self.external:
            names = ", ".join(f"{BOTS[b.bot].label} (PID {b.pid})" for b in self.external)
            self.state_label.setText(dot(ORANGE, f"lancé hors du panneau : {names}"))
        else:
            self.state_label.setText(dot(GREY, "arrêté"))
        if running and self.started_at is not None:
            seconds = int(time.monotonic() - self.started_at)
            self.uptime_label.setText(f"{seconds // 3600:d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}")
        stats = self.stats
        self.matches_label.setText(f"{stats.queued} lancé(s), {stats.finished} terminé(s)")
        self.plays_label.setText(str(stats.plays))
        self.errors_label.setText(f"{stats.errors}  {stats.last_error[-80:]}" if stats.errors else "0")
        self.event_label.setText(stats.last_event[-120:] or "—")
        self.start_button.setEnabled(not running and not self.external)
        self.after_button.setEnabled(running and not self.stop_after_match)
        self.stop_button.setEnabled(running or bool(self.external))

    def _on_preview_toggled(self, enabled: bool) -> None:
        if not enabled:
            self.preview.show_message("Aperçu désactivé")

    def _request_preview(self) -> None:
        if not self.preview_check.isChecked():
            return
        if not self.device.state.online:
            self.preview.show_message("Émulateur injoignable")
            return
        busy = self.runner.running or bool(self.external)
        self.device.set_busy(busy)
        if busy and time.monotonic() - self.last_preview < PREVIEW_BUSY_SECONDS:
            return
        if self.vision_check.isChecked():
            submitted = self._shot.submit(grab_and_analyse, self.vision, self.device.adb(),
                                          self.device.serial(), self._vision_deck())
        else:
            submitted = self._shot.submit(grab_image, self.device.adb(), self.device.serial())
        if submitted:
            self.last_preview = time.monotonic()

    def _vision_deck(self) -> str:
        """The deck the selected bot reads its hand against: the native presets or Hog 2.6."""
        spec = BOTS[self.bot_combo.currentData()]
        return self.deck_combo.currentData() if spec.uses_deck else "hog26"

    def _on_shot(self, result) -> None:
        if not self.preview_check.isChecked() or result is None:
            return
        if isinstance(result, Exception):
            self.vision_label.setText(f"Vision indisponible : {result}")
            return
        if isinstance(result, tuple):
            image, seen = result
            self.vision_label.setText(summary(seen))
        else:
            image = result
            self.vision_label.setText("—")
        self.preview.show_image(image)

    def closeEvent(self, event) -> None:      # noqa: N802 - Qt naming
        self._save_settings()
        if self.runner.running:
            answer = QMessageBox.question(
                self, "Bot en cours",
                "Un bot lancé depuis le panneau tourne encore. L'arrêter et fermer ?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.runner.stop()
        self._restore_screen()
        event.accept()

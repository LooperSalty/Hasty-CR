@echo off
rem "Vision du bot": a window docked beside MuMu showing what the bot sees,
rem its elixir, its hand and why it plays what it plays.
cd /d "%~dp0scripts"
start "" "%~dp0.venvs\buildabot\Scripts\pythonw.exe" -m control_panel.live_view

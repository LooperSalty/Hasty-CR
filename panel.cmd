@echo off
rem Double-clickable launcher for the bot control panel.
rem Pick a bot, start it, watch it play, stop it - no terminal needed.
cd /d "%~dp0scripts"
start "" "%~dp0.venvs\buildabot\Scripts\pythonw.exe" -m control_panel

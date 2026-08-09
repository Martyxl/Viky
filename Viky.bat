@echo off
REM Double-click to launch Viky (web UI). Shows a console with model-loading logs.
REM For a no-console launch, change python.exe to pythonw.exe below and make a
REM shortcut to this file (right-click .bat -> Send to -> Desktop).
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" scripts\viky_ui.py
pause

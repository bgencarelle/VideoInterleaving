@echo off
rem Windows twin of ./vi.scope-gui: runs it with this folder's .venv Python.
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\scope_launcher_gui.py" %*

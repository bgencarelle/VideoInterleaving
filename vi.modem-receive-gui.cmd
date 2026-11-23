@echo off
rem Windows twin of ./vi.modem-receive-gui: runs it with this folder's .venv Python.
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\v7_live.py" gui %*

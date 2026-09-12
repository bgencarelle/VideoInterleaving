@echo off
setlocal
cd /d "%~dp0"
for %%A in (%*) do if "%%~A"=="--non-interactive" set "MODEM_NO_PAUSE=1"
if defined MODEM_PYTHON goto explicit
py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1
if not errorlevel 1 goto pylauncher
python -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1
if not errorlevel 1 goto pythonlauncher
goto missing
:explicit
"%MODEM_PYTHON%" -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1
if errorlevel 1 goto missing
if "%~1"=="" ("%MODEM_PYTHON%" setup_decode.py --install) else ("%MODEM_PYTHON%" setup_decode.py %*)
goto done
:pylauncher
if "%~1"=="" (py -3 setup_decode.py --install) else (py -3 setup_decode.py %*)
goto done
:pythonlauncher
if "%~1"=="" (python setup_decode.py --install) else (python setup_decode.py %*)
goto done
:missing
echo Python 3.11+ was not found. Install Python with Tcl/Tk and pip enabled. 1>&2
echo Set MODEM_PYTHON to the full executable path if Python is already installed. 1>&2
echo Python installers: https://www.python.org/downloads/windows/ 1>&2
echo Optional WinGet / App Installer: https://learn.microsoft.com/en-us/windows/package-manager/winget/ 1>&2
if "%MODEM_NO_PAUSE%"=="1" (endlocal & exit /b 1)
choice /c YN /n /m "Open Python installation instructions? [Y/N] "
if errorlevel 2 goto missingdone
start "" "https://www.python.org/downloads/windows/"
:missingdone
pause
endlocal & exit /b 1
:done
set "setup_result=%errorlevel%"
if not "%setup_result%"=="0" echo Decoder setup failed (exit %setup_result%). 1>&2
if not "%MODEM_NO_PAUSE%"=="1" pause
endlocal & exit /b %setup_result%

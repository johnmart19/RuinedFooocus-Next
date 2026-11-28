@echo off
setlocal
echo Starting RuinedFooocus...
cd /d "%~dp0"

rem Respect explicitly selected and activated environments.
if defined PYTHON goto explicit_python
if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" goto active_python
if exist "venv\Scripts\python.exe" goto local_python

python -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>&1
if not errorlevel 1 goto create_with_python
py -3 -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>&1
if not errorlevel 1 goto create_with_launcher
echo Python 3.10 or newer was not found or could not start.
echo Install Python from https://www.python.org/downloads/windows/ and reopen your terminal.
echo Alternatively, set PYTHON to the full path of a working Python executable.
exit /b 1

:create_with_python
python -m venv venv
if errorlevel 1 goto environment_failed
goto local_python

:create_with_launcher
py -3 -m venv venv
if errorlevel 1 goto environment_failed
goto local_python

:explicit_python
set "RF_PYTHON=%PYTHON%"
goto run

:active_python
set "RF_PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
goto run

:local_python
set "RF_PYTHON=%~dp0venv\Scripts\python.exe"

:run
"%RF_PYTHON%" -u entry_with_update.py %*
exit /b %errorlevel%

:environment_failed
echo Could not create venv. Check your Python installation and folder permissions.
exit /b 1

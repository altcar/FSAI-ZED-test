@echo off
setlocal
cd /d "%~dp0"
echo Install Python 3.11 x64 from python.org first. Internet is required for dependencies.
py -3.11 -m venv .venv
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip freeze > installed_requirements.txt
".venv\Scripts\python.exe" scripts\check_environment.py
if errorlevel 1 goto fail
echo Setup complete. Run 01_verify_and_replay.cmd, then 02_run_depthnet.cmd.
pause
exit /b 0
:fail
echo Setup FAILED. Keep this window open and read the error above. Do not continue.
pause
exit /b 1

@echo off
setlocal
cd /d "%~dp0"
set "VENV_DIR=C:\FSAI_3070_venv"
echo Setting up Python 3.11 environment...
if not exist "%VENV_DIR%\Scripts\python.exe" (
  where uv >nul 2>&1
  if not errorlevel 1 (
    uv venv "%VENV_DIR%" --python 3.11
  ) else (
    py -3.11 -m venv "%VENV_DIR%"
  )
)
if errorlevel 1 goto fail
if not exist .venv (
  cmd /c mklink /J .venv "%VENV_DIR%"
)
if not exist .venv\Scripts\python.exe goto fail
set "PY=.venv\Scripts\python.exe"
"%PY%" -m ensurepip --upgrade
if errorlevel 1 goto fail
where uv >nul 2>&1
if not errorlevel 1 (
  uv pip install --python "%PY%" torch torchvision --index-url https://download.pytorch.org/whl/cu128
  if errorlevel 1 goto fail
  uv pip install --python "%PY%" -r requirements.txt
) else (
  "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  if errorlevel 1 goto fail
  "%PY%" -m pip install -r requirements.txt
)
if errorlevel 1 goto fail
"%PY%" scripts\check_environment.py
if errorlevel 1 goto fail
echo Setup complete. Run 01_verify_and_replay.cmd, then 02_run_depthnet.cmd.
pause
exit /b 0
:fail
echo Setup FAILED. Keep this window open and read the error above. Do not continue.
pause
exit /b 1
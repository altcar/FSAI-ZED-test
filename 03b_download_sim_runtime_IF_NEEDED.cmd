@echo off
setlocal
cd /d "%~dp0"
echo Optional official simulator runtime download. Read ZED_README.md first.
".venv\Scripts\python.exe" scripts\download_sim_runtime.py
if errorlevel 1 goto fail
echo Download complete. Runtime ABI is checked when 04 runs.
pause
exit /b 0
:fail
echo Download failed. Preserve the error above.
pause
exit /b 1

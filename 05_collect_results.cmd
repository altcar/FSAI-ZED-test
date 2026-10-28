@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\collect_results.py
if errorlevel 1 goto fail
pause
exit /b 0
:fail
echo Could not collect results. Read the error above.
pause
exit /b 1

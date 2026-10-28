@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\compare.py --device 0 --instances data\mono_demo\instances_pinhole.png --output outputs\mono
if errorlevel 1 goto fail
echo Success. See outputs\mono\summary.json and comparison.png.
pause
exit /b 0
:fail
echo FAILED. Run scripts\check_environment.py with the kit Python and check CUDA.
pause
exit /b 1

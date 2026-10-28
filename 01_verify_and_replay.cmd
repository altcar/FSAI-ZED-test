@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\verify_manifest.py
if errorlevel 1 goto fail
".venv\Scripts\python.exe" scripts\compare.py --prediction reference_results\single_depth_check.prediction.npy --instances data\mono_demo\instances_pinhole.png --output outputs\replay
if errorlevel 1 goto fail
echo Success. See outputs\replay\summary.json and comparison.png.
pause
exit /b 0
:fail
echo FAILED. Read the error above.
pause
exit /b 1

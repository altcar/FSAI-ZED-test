@echo off
setlocal
cd /d "%~dp0"
echo EXPERIMENTAL: Windows SDK bridge has not been GPU-tested on the packaging Mac.
echo Requires SDK 5.5.x and matching sim-streamer exports. See ZED_README.md.
".venv\Scripts\python.exe" scripts\run_sdk.py
if errorlevel 1 goto fail
echo Success. See outputs\sdk_three_way\comparison\summary.json and comparison.png.
pause
exit /b 0
:fail
echo SDK bridge FAILED. Save the error and outputs\sdk_three_way\streamer.log if present.
echo Do not report this run as a completed three-way comparison.
pause
exit /b 1

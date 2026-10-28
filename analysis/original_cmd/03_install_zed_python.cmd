@echo off
setlocal
cd /d "%~dp0"
set "FSAI_ZED_DIR=%ZED_SDK_ROOT_DIR%"
if not defined FSAI_ZED_DIR set "FSAI_ZED_DIR=C:\Program Files (x86)\ZED SDK"
if not exist "%FSAI_ZED_DIR%\get_python_api.py" goto missing
".venv\Scripts\python.exe" "%FSAI_ZED_DIR%\get_python_api.py"
if errorlevel 1 goto fail
".venv\Scripts\python.exe" scripts\check_environment.py
if errorlevel 1 goto fail
echo Python API installed. Read ZED_README.md before the experimental bridge.
pause
exit /b 0
:missing
echo First install ZED SDK 5.5.x for Windows CUDA 12 from https://www.stereolabs.com/developers/release
echo Set ZED_SDK_ROOT_DIR if installed in a custom folder.
:fail
echo ZED setup did not complete. Read the message above.
pause
exit /b 1

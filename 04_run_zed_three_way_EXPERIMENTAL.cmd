@echo off
setlocal
cd /d "%~dp0"
echo FIXED Windows SDK bridge: PERFORMANCE, H.264 3000, chunk 1200, transport BOTH.
if not defined ZED_SDK_ROOT_DIR (
  if exist "L:\save\userdata\Downloads\ZED_SDK_extract\app" (
    set "ZED_SDK_ROOT_DIR=L:\save\userdata\Downloads\ZED_SDK_extract\app"
  ) else if exist "C:\Program Files (x86)\ZED SDK" (
    set "ZED_SDK_ROOT_DIR=C:\Program Files (x86)\ZED SDK"
  )
)
if not defined ZED_SDK_ROOT_DIR goto nosdk
if not exist zed_runtime\sl_zed64.dll goto nodll
if not exist .venv\Scripts\python.exe goto novenv
".venv\Scripts\python.exe" scripts\run_sdk_fixed.py --depth-mode PERFORMANCE --output outputs\sdk_three_way
if errorlevel 1 goto fail
".venv\Scripts\python.exe" analysis\fix_sdk_left.py
if errorlevel 1 goto fail
".venv\Scripts\python.exe" scripts\compare.py --image outputs\sdk_three_way\sdk_left_corrected.png --gt outputs\sdk_three_way\ue_z_m.npy --sdk-depth outputs\sdk_three_way\sdk_z_m.npy --device 0 --output outputs\sdk_three_way\comparison_corrected
if errorlevel 1 goto fail
echo Success. See outputs\sdk_three_way\comparison_corrected\summary.json.
echo Update FSAI_STEREO_DEPTH_REPORT.md if the new metrics differ.
pause
exit /b 0
:nosdk
echo ZED SDK root not found. Run 03_install_zed_python.cmd or set ZED_SDK_ROOT_DIR.
goto fail
:nodll
echo zed_runtime\sl_zed64.dll missing. Run 03b_download_sim_runtime_IF_NEEDED.cmd.
goto fail
:novenv
echo .venv missing. Run 00_setup_windows.cmd.
:fail
echo SDK bridge FAILED. Preserve the error and outputs\sdk_three_way\streamer.log.
pause
exit /b 1
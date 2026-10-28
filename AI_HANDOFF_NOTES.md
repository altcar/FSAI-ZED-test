# AI Handoff Notes — Read Before Diagnosing

**Do not repeat the diagnoses below.** They are already proven on this machine
(RTX 3070 Laptop, Windows 11, driver **621.01**). The working result and the
exact failure causes are in `RUN_SUMMARY.md`,
`STEREO_DEPTH_VALIDATION.md`, `ZED_SDK_SIMULATION_RESULT.md`, and the executed
`FSAI_STEREO_RESULTS.ipynb`.

## State after this run

- Git repo is initialized in this folder.
  - `e6c1af8` — original ZIP import.
  - `ab18af9` — validated state before the final rerun.
  - `939bc4e` — final rerun + notebook + summary + these notes.
  - HEAD — fixed `00`/`03`/`04` entry points and the fixed rerun.
- **Dependencies were removed after the final commit**:
  `C:\FSAI_3070_venv`, the `.venv` junction, `L:\...\ZED_SDK_extract`,
  `zed_runtime\sl_zed64.dll`, downloaded wheels/installer, and
  `C:\Users\V9\innoextract`. There is no runnable venv/SDK in the workspace.
  This is intentional; the fixed `00` rebuilds the venv.
- Results/figures/logs/reports remain in `outputs/` and the Markdown files.
- `scripts/verify_manifest.py` checks 461 files. The fixed `00`, `03`, `04`
  hashes were updated in `MANIFEST_SHA256.json`; the original copies are in
  `analysis/original_cmd/`.

## Do-not-repeat list (already diagnosed and fixed)

1. **Driver/NVENC.** The machine started on driver 555.97 (NVENC 12.2). The SDK
   needs 610+; installing 621.01 fixed NVENC. The install initially failed with
   `Error 112: There is not enough space on the disk` during DriverStore import;
   that was disk space, not a bad package. Do not re-diagnose 555.97/NVENC.
2. **The 5.5.1 runtime does not exist on the CDN.** The official `build.bat`
   pins `SL_ZED_VERSION=5.2.0`; `.../sl_zed/5.5.1/...` returns 403. The 5.2.0
   archive reports SDK 5.5.1 but sends **1,216-byte** camera metadata; the
   5.5.1 consumer expects **2,904** and skips it:
   `Incorrect metadata size. Expecting: 2904 got 1216. Skipping.`
   K and baseline still pass. Do not download another "5.5.1 runtime".
3. **Stock original 04 fails; the current fixed 04 passes.** The original
   `04_run_zed_three_way_EXPERIMENTAL.cmd` (preserved in
   `analysis/original_cmd/`) used the stock RGB/`chunk_size=4096` producer and
   fails with `ENCODING_FAILED` / metadata timeout / `int(ERROR_CODE)`
   TypeError. The current `04` uses `scripts/run_sdk_fixed.py` and
   `scripts/sim_streamer_fixed.py`; it passes. Do not restore the stock 04.
4. **The fix is known.** Use:
   - `chunk_size = 1200`;
   - extended `StreamingParameters` fields after `chunk_size`:
     `adaptative_bitrate`, `enable_downscale`, `pad_extra[2]`,
     `target_width=width`, `target_height=height`, `target_bitrate=3000`,
     then `gpu_input`, `stream_depth`, depth fields;
   - H.264, 3000 kbps, transport BOTH (2), frame-relative timestamps;
   - tolerate the first `ENCODING_FAILED` for the first 10 frames (issue #61);
   - depth mode **PERFORMANCE**.
   The canonical fixed files are `scripts/sim_streamer_fixed.py` and
   `scripts/run_sdk_fixed.py`; copies of the development versions remain in
   `analysis/sim_streamer_success_experimental.py` and
   `analysis/run_sdk_success_experimental.py`.
5. **NEURAL is not available yet.** The extracted SDK's TensorRT DLLs
   (`nvinfer_10`, `nvinfer_lean_10`, `nvinfer_plugin_10`, `nvonnxparser_10`,
   `nvinfer_dispatch_10`, `nvinfer_vc_plugin_10`, `zlibwapi`) were not on the
   consumer search path, so NEURAL stops with `NEURAL TRT NOT FOUND`. Do not
   label PERFORMANCE as NEURAL. To attempt NEURAL, copy those DLLs into
   `site-packages/pyzed` and retest.
6. **The saved SDK left image has an R/B swap.** `sdk_left.png` scores 20.54 dB
   against the prepared left; after R/B swap it is 28.80 dB. DepthNet on the
   swapped file scores 9.34 m — that number is invalid. Always use
   `sdk_left_corrected.png` for DepthNet and the fair comparison. The SDK depth
   itself is unchanged.
7. **Run-to-run variation.** Two identical PERFORMANCE captures gave SDK MAE
   2.800 m and 3.117 m (coverage 92.74% and 92.33%). This is H.264/IDR/duplicate
   timing, not a deterministic benchmark. The final fixed-04 report uses 2.800 m.

## Expected values (do not re-litigate)

| Check | Expected |
|---|---|
| Manifest | 461/461 |
| Unit tests | 4/4 |
| Replay | 419,524 px; MAE 2.613193 m; RMSE 2.747712 m |
| Mono GPU | 419,524 px; MAE 2.610419 m; RMSE 2.745033 m |
| Stereo DepthNet clean | MAE 2.530 m; RMSE 2.718 m; 100% |
| Stereo SGBM clean | MAE 0.791 m; RMSE 4.301 m; 73.0% |
| SDK PERFORMANCE (final) | 251,213 px; 92.74%; MAE 2.800 m; RMSE 6.879 m |
| DepthNet on corrected SDK image | 270,876 px; 100%; MAE 2.211 m; RMSE 2.464 m |
| Common pixels | 251,213 |

## If a rerun is required

The dependencies were removed after the final commit. The fixed
`00_setup_windows.cmd` now rebuilds the venv even though this PC has no
`py -3.11` launcher entry: it uses `uv` (or `py -3.11` as fallback), creates
`C:\FSAI_3070_venv`, junctions `.venv`, installs CUDA PyTorch and requirements.

1. Restore the SDK/runtime assets:
   - SDK 5.5.1: extract the official installer to
     `L:\save\userdata\Downloads\ZED_SDK_extract` (installer was removed after
     the run). `get_python_api.py`, `include/sl/Camera.hpp`,
     `bin/sl_zed64.dll`, `bin/sl_ai64.dll` are required.
   - Sim runtime: download the 5.2.0 archive from the URL in
     `scripts/download_sim_runtime.py`; extract **twice** (the `.tar.gz`
     contains a `.tar`) and keep `zed_runtime/sl_zed64.dll`.
2. Run:
   ```powershell
   .\00_setup_windows.cmd
   $env:ZED_SDK_ROOT_DIR = 'L:\save\userdata\Downloads\ZED_SDK_extract\app'
   .\03_install_zed_python.cmd
   .\03b_download_sim_runtime_IF_NEEDED.cmd   # only if the DLL is absent
   .\04_run_zed_three_way_EXPERIMENTAL.cmd
   .\05_collect_results.cmd
   ```
   The fixed `04` runs `scripts/run_sdk_fixed.py` +
   `scripts/sim_streamer_fixed.py`, fixes the saved left image's R/B swap,
   compares on the corrected image, and rebuilds the SDK report. Do not copy
   the experimental files over the canonical scripts; the fixed scripts are
   already the canonical files.
3. Rebuild the stereo/SDK Markdown and notebook if needed:
   ```powershell
   .venv\Scripts\python.exe analysis\build_report.py
   .venv\Scripts\python.exe analysis\build_sdk_success_report.py
   uv pip install --python .venv\Scripts\python.exe nbformat nbclient ipykernel
   .venv\Scripts\python.exe -m ipykernel install --user --name fsai --display-name "FSAI Results"
   .venv\Scripts\python.exe analysis\make_notebook.py
   ```

If the SDK open reports metadata 1216 vs 2904, that is expected for the
5.2.0-labelled runtime; it does not invalidate the depth as long as K/baseline
pass and the capture has valid pixels. Do not remove the version guard.

## What not to do

- Do not retry driver/NVENC/disk-space diagnosis; it is solved (621.01).
- Do not run the stock original `04` from `analysis/original_cmd/`; use the
  current fixed `04`.
- Do not use `sdk_left.png` (R/B swapped) for DepthNet; use the corrected file.
- Do not call PERFORMANCE "NEURAL"; the report labels it.
- Do not commit `.venv`, SDK trees, runtime DLLs, wheels, `.npy` arrays, or
  bridge temp dirs; `.gitignore` excludes them.
- Do not report the invalid 9.34 m DepthNet or the 0-valid-pixel capture as the
  result; those were intermediate failures.
- Do not re-add the old `00`/`03`/`04` content from the ZIP; the fixed commands
  and updated manifest are intentional.

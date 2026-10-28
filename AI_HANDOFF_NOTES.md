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
  - HEAD — final rerun + notebook + summary + these notes.
- **Dependencies were removed after the final commit**:
  `C:\FSAI_3070_venv`, the `.venv` junction, `L:\...\ZED_SDK_extract`,
  `zed_runtime\sl_zed64.dll`, downloaded wheels, and `C:\Users\V9\innoextract`.
  There is no runnable venv/SDK in the workspace. This is intentional.
- Results/figures/logs/reports remain in `outputs/` and the Markdown files.
- `scripts/verify_manifest.py` checks the original 461 files; the new reports,
  analysis scripts, notebook and outputs are not in the original manifest.

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
3. **Stock producer fails.** Stock `04_run_zed_three_way_EXPERIMENTAL.cmd`
   (RGB/RGBA, `chunk_size=4096`, strict first-frame handling) fails with
   `ENCODING_FAILED` / metadata timeout / `int(ERROR_CODE)` TypeError. This is
   expected. Use the fixed experimental producer, not stock 04.
4. **The fix is known.** Use:
   - `chunk_size = 1200`;
   - extended `StreamingParameters` fields after `chunk_size`:
     `adaptative_bitrate`, `enable_downscale`, `pad_extra[2]`,
     `target_width=width`, `target_height=height`, `target_bitrate=3000`,
     then `gpu_input`, `stream_depth`, depth fields;
   - H.264, 3000 kbps, transport BOTH (2), frame-relative timestamps;
   - tolerate the first `ENCODING_FAILED` for the first 10 frames (issue #61);
   - depth mode **PERFORMANCE**.
   The working files are preserved as
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
   timing, not a deterministic benchmark. The final report uses 3.117 m.

## Expected values (do not re-litigate)

| Check | Expected |
|---|---|
| Manifest | 461/461 |
| Unit tests | 4/4 |
| Replay | 419,524 px; MAE 2.613193 m; RMSE 2.747712 m |
| Mono GPU | 419,524 px; MAE 2.610419 m; RMSE 2.745033 m |
| Stereo DepthNet clean | MAE 2.530 m; RMSE 2.718 m; 100% |
| Stereo SGBM clean | MAE 0.791 m; RMSE 4.301 m; 73.0% |
| SDK PERFORMANCE (final) | 250,089 px; 92.33%; MAE 3.117 m; RMSE 7.930 m |
| DepthNet on corrected SDK image | 270,876 px; 100%; MAE 2.312 m; RMSE 2.548 m |
| Common pixels | 250,089 |

## If a rerun is required

You must rebuild the removed dependencies first. Do not expect `00_setup` to
work: this PC has **no `py -3.11`** launcher entry, so `00_setup_windows.cmd`
fails. Use uv:

```powershell
uv venv C:\FSAI_3070_venv --python 3.11
uv pip install --python C:\FSAI_3070_venv\Scripts\python.exe torch torchvision `
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python C:\FSAI_3070_venv\Scripts\python.exe -r requirements.txt
# link it as .venv (or run C:\FSAI_3070_venv directly)
```

Then restore the SDK/runtime assets:

- SDK 5.5.1: extract the official installer to
  `L:\save\userdata\Downloads\ZED_SDK_extract` (the official installer was
  removed after the run).
- `pyzed` wheel: `pyzed-5.5-cp311-cp311-win_amd64.whl` into the venv; copy
  `sl_zed64.dll` and `sl_ai64.dll` from the SDK `bin` into
  `site-packages/pyzed`.
- Sim runtime: download the 5.2.0 archive from the URL in
  `scripts/download_sim_runtime.py` and extract `sl_zed64.dll` into
  `zed_runtime/`.
- Run the fixed producer:
  ```powershell
  Copy-Item analysis\sim_streamer_success_experimental.py scripts\sim_streamer.py -Force
  Copy-Item analysis\run_sdk_success_experimental.py scripts\run_sdk.py -Force
  $env:ZED_SDK_ROOT_DIR = 'L:\save\userdata\Downloads\ZED_SDK_extract\app'
  .venv\Scripts\python.exe analysis\run_sdk_with_root.py `
    --dll zed_runtime\sl_zed64.dll --output outputs\sdk_three_way --depth-mode PERFORMANCE
  .venv\Scripts\python.exe analysis\fix_sdk_left.py
  .venv\Scripts\python.exe scripts\compare.py `
    --image outputs\sdk_three_way\sdk_left_corrected.png `
    --gt outputs\sdk_three_way\ue_z_m.npy `
    --sdk-depth outputs\sdk_three_way\sdk_z_m.npy `
    --device 0 --output outputs\sdk_three_way\comparison_corrected
  .venv\Scripts\python.exe analysis\build_sdk_success_report.py
  # restore the canonical kit scripts afterwards
  Copy-Item analysis\sim_streamer_original.py scripts\sim_streamer.py -Force
  Copy-Item analysis\run_sdk_original.py scripts\run_sdk.py -Force
  ```

If the SDK open reports metadata 1216 vs 2904, that is expected for the
5.2.0-labelled runtime; it does not invalidate the depth as long as K/baseline
pass and the capture has valid pixels. Do not remove the version guard.

## What not to do

- Do not retry driver/NVENC/disk-space diagnosis; it is solved (621.01).
- Do not use the stock `04` bridge as-is; it fails by design.
- Do not use `sdk_left.png` (R/B swapped) for DepthNet; use the corrected file.
- Do not call PERFORMANCE "NEURAL"; the report labels it.
- Do not commit `.venv`, SDK trees, runtime DLLs, wheels, `.npy` arrays, or
  bridge temp dirs; `.gitignore` excludes them.
- Do not report the invalid 9.34 m DepthNet or the 0-valid-pixel capture as the
  result; those were intermediate failures.

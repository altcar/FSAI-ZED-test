# FSAI 3070 Stereo Depth — Run Summary

**Date:** 2026-10-28  
**Hardware:** RTX 3070 Laptop 8 GB · NVIDIA driver **621.01** · Windows 11  
**SDK:** ZED SDK **5.5.1** · `pyzed` **5.5.1** · official sim runtime **5.5.1** (5.2.0-labelled archive)  
**Final SDK mode:** **PERFORMANCE** · H.264 3000 kbps · chunk 1200 · transport BOTH

This is the command/test record for the final rerun. The full evidence is in
`STEREO_DEPTH_VALIDATION.md` and `ZED_SDK_SIMULATION_RESULT.md`; the executed
notebook is `FSAI_STEREO_RESULTS.ipynb`.

## Git versions

| Version | Commit | Content |
|---|---|---|
| 1 | `e6c1af8` | Original ZIP import (`FSAI_3070_Windows_Kit.zip`) |
| 2 | `ab18af9` | Current validated state before final rerun |
| 3 | `939bc4e` | Final rerun, notebook, summary, AI handoff notes |
| 4 | HEAD | Fixed `00`/`03`/`04` entry points and reran them |

## Kit entry points

| Entry point | Result | Expected / notes |
|---|---|---|
| `00_setup_windows.cmd` | **PASS (fixed)** | Uses `uv` when `py -3.11` is unavailable; creates `C:\FSAI_3070_venv` + `.venv` junction, installs CUDA PyTorch and requirements; CUDA check passes. |
| `01_verify_and_replay.cmd` | **PASS** | 461/461 manifest; replay 419,524 px; MAE **2.613193 m**; RMSE **2.747712 m**; output `outputs/replay`. |
| `02_run_depthnet.cmd` | **PASS** | Fresh mono GPU: 419,524 px; MAE **2.610419 m**; RMSE 2.745033 m; output `outputs/mono`. |
| `03_install_zed_python.cmd` | **PASS (fixed)** | `pyzed` 5.5.1 installed; the helper's numpy 2.4.6 is re-pinned to **2.2.6** so opencv 4.12 is valid; SDK 5.5.1. |
| `03b_download_sim_runtime_IF_NEEDED.cmd` | **EXPECTED REFUSAL** | `zed_runtime` already contains `sl_zed64.dll`; the guard refuses to overwrite. |
| `04_run_zed_three_way_EXPERIMENTAL.cmd` | **PASS (fixed)** | Uses `scripts/run_sdk_fixed.py` + `scripts/sim_streamer_fixed.py`, PERFORMANCE mode, corrected-color comparison and report build. The original stock `04` is preserved in `analysis/original_cmd/`. |
| `05_collect_results.cmd` | **PASS** | Creates `FSAI_results_<UTC>.zip` from `outputs/`. |

The fixed `.cmd` files are intentional; `MANIFEST_SHA256.json` was updated for
`00`, `03` and `04`, so `verify_manifest.py` still reports 461/461. The original
copies remain in `analysis/original_cmd/`.

## Direct tests

| Test | Result |
|---|---|
| `scripts/verify_manifest.py` | PASS — 461 original files |
| `python -m unittest discover -s tests -v` | PASS — 4/4 |
| Replay to `outputs/replay_windows` | PASS — 419,524 px; MAE 2.613193 m; RMSE 2.747712 m |
| Fresh mono GPU to `outputs/mono_gpu` | PASS — MAE 2.610419 m; RMSE 2.745033 m |
| Stereo validation (`analysis/stereo_depth_validation.py`) | PASS — see below |
| Fixed SDK capture (`analysis/run_sdk_with_root.py --depth-mode PERFORMANCE`) | PASS — see below |
| Fair compare on corrected SDK left | PASS — `outputs/sdk_three_way/comparison_corrected/summary.json` |
| `analysis/check_reports.py` | PASS — tables and image links valid |

## Final depth results

### Clean stereo pair (0 < Z ≤ 20 m, common pixels)

| Method | Coverage | MAE | RMSE | Bias | Median | P90 |
|---|---:|---:|---:|---:|---:|---:|
| DepthNet (clean prepared left) | 100% | **2.530 m** | 2.718 m | +1.970 m | 2.484 m | 2.998 m |
| OpenCV SGBM (rectified pair) | 73.00% | **0.791 m** | 4.301 m | +0.453 m | 0.013 m | 0.549 m |

Geometry: f = 636.25 px; B = 119.7245 mm; epipolar median |dy| 0.117 px;
SGBM median |Δd| 0.265 px; 89.5% within 1 px.

### ZED SDK PERFORMANCE capture (final run)

| Method | Valid px | Coverage | MAE | RMSE | Bias | Median | P90 | ≥3 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ZED SDK | 251,213 | 92.74% | **2.800 m** | 6.879 m | +2.725 m | 1.631 m | 2.910 m | 9.7% |
| DepthNet on corrected SDK image | 270,876 | 100% | **2.211 m** | 2.464 m | +1.567 m | 2.122 m | 2.725 m | 7.1% |

Common pixels: **251,213**. DepthNet wins MAE/RMSE and coverage; the SDK has the
lower median but a heavier tail. Two identical PERFORMANCE captures gave MAE
**2.800 m** and **3.117 m**; this summary/report use the fixed-04 run at 2.800 m.
The variation is H.264/IDR/duplicate-frame timing, not a deterministic benchmark.

The original `sdk_left.png` has an R/B channel swap; DepthNet on it scores
9.34 m and is **invalid**. The fair comparison uses
`outputs/sdk_three_way/sdk_left_corrected.png` (PSNR 28.80 dB vs prepared left).

## Why the original stock SDK path failed (fixed in `04`)

- Stereolabs issue **#63**: the bundled runtime needs extra `StreamingParameters`
  fields (`target_width`, `target_height`, `target_bitrate`, adaptive/downscale),
  `chunk_size ≤ 1200`, H.264 fallback, and explicit GPU context.
- Stereolabs issue **#61**: first-frame `ENCODING_FAILED` is a known bug fixed in
  5.3; it must not be fatal.
- `build.bat` pins `SL_ZED_VERSION=5.2.0`; the receiver log shows
  `Incorrect metadata size. Expecting: 2904 got 1216. Skipping.` — the 5.2.0
  producer and 5.5.1 consumer do not share camera metadata. K/baseline still pass.
- The current fixed `04` uses `scripts/run_sdk_fixed.py` and
  `scripts/sim_streamer_fixed.py`; the original stock files are preserved in
  `analysis/original_cmd/`.

## Cleanup (performed after the final commit)

Removed:

- `C:\FSAI_3070_venv` (venv target)
- `.venv` junction in the kit
- `L:\save\userdata\Downloads\ZED_SDK_extract`
- `zed_runtime\sl_zed64.dll` (downloaded runtime binary)
- downloaded `*.whl` files in the kit
- `C:\Users\V9\innoextract`
- any SDK/runtime installer archives

Kept: `outputs/` results, figures, logs, reports, notebook, analysis scripts,
original kit files, and the git history.

To rerun after cleanup, rebuild the venv/SDK per `AI_HANDOFF_NOTES.md`.

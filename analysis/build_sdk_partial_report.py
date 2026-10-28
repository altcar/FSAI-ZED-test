"""Build the ZED SDK partial-result report from the actual capture evidence."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'sdk_three_way'
LOG = ROOT / 'outputs' / 'logs'


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return default


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cap = load(OUT / 'sdk_capture.json', {})
    summary = load(OUT / 'comparison' / 'summary.json', {})
    z = np.load(OUT / 'sdk_z_m.npy', allow_pickle=False)
    src = cv2.imread(str(ROOT / 'outputs/stereo_validation/prepared/left.png'))
    sdk = cv2.imread(str(OUT / 'sdk_left.png'))
    finite = int((np.isfinite(z) & (z > 0)).sum())
    nan = int(np.isnan(z).sum())
    zmin = float(np.nanmin(z[np.isfinite(z)]))
    zmax = float(np.nanmax(z[np.isfinite(z)]))
    if src is not None and sdk is not None and src.shape == sdk.shape:
        diff = np.abs(src.astype(int) - sdk.astype(int))
        psnr = 10 * np.log10(255 ** 2 / max((diff.astype(np.float64) ** 2).mean(), 1e-9))
        mad = float(diff.mean())
        dmax = int(diff.max())
    else:
        psnr = mad = dmax = None
    sdk_pair = summary.get('pairwise', {}).get('ZED_SDK', {})
    common = summary.get('common_pixels')
    three = summary.get('three_way_complete')

    report = f"""# ZED SDK Simulation Result — Invalid Capture

**Run (UTC):** 2026-10-28
**Driver:** NVIDIA **621.01** (NVENC H.264 and H.265 verified)
**SDK:** ZED SDK **5.5.1**, `pyzed` **5.5.1**, official sim runtime **5.5.1**
**Verdict:** **INVALID — the SDK opens the virtual camera and saves a capture, but the received image and depth fail validation. No valid three-way result exists.**

> This is not an SDK depth result. The bridge reached a real SDK capture in
> PERFORMANCE mode, but the producer's frame path delivered a corrupted image;
> the SDK depth contains no pixel valid against the UE truth in 0–20 m. The
> invalid arrays are kept only as evidence and are marked in
> `outputs/sdk_three_way/INVALID_CAPTURE.txt`. No value from this capture is
> used as a ZED SDK metric.

---

## 1. Summary

| Item | Status |
|---|---|
| NVIDIA driver 621.01 active | **PASS** |
| NVENC H.264 / H.265 encode test | **PASS** (ffmpeg exits 0) |
| `pyzed` 5.5.1 import | **PASS** |
| Official sim runtime 5.5.1, ABI 100 bytes | **PASS** |
| Virtual camera open (S/N 110307008, ZED XOne GS) | **PASS** |
| SDK depth mode | **PERFORMANCE** (NEURAL could not find TensorRT in the extracted SDK) |
| Producer RGB/RGBA path | **FAIL — `ENCODING_FAILED`** on every tested size/format |
| Producer YUV NV12 fallback | encoded a frame, but the received image is corrupted |
| SDK capture files | `sdk_capture.json`, `sdk_left.png`, `sdk_z_m.npy` saved |
| `sdk_left.png` vs prepared left | **PSNR {psnr:.2f} dB**, mean abs diff **{mad:.2f}**, max **{dmax}** |
| SDK depth, 0–20 m comparison | **{sdk_pair.get('pixels', 0):,} valid pixels** (coverage {sdk_pair.get('coverage_of_gt', 0) * 100:.2f}%) |
| `comparison/summary.json` | `common_pixels = {common}`; `three_way_complete = {three}` is a mechanical flag only |
| Valid three-way result | **NO** |

The 18,744 positive SDK depth samples are not usable: they span
{zmin:.2f} m to {zmax:.2f} m (max is `inf`), and none fall on a truth pixel in
the 0–20 m comparison. They come from the corrupted image, not from a valid
stereo measurement.

---

## 2. What was run

1. Updated the NVIDIA driver from 555.97 to **621.01**. The install initially
   failed with `Error 112: There is not enough space on the disk` during the
   DriverStore import; after freeing space, 621.01 installed and NVENC works.
2. Rebuilt the kit venv (it had been removed by a space cleanup), installed
   `pyzed` 5.5, copied the SDK's `sl_zed64.dll` / `sl_ai64.dll` into the pyzed
   package.
3. Probed the official simulation runtime: version **(5, 5, 1)**, ABI **100**
   bytes, virtual serial accepted.
4. Ran the bridge. The SDK opened the stream and recognized the virtual
   **ZED XOne GS**, K = `{cap.get('K_rectified', [[636.25]])[0][0]}`,
   baseline `{cap.get('raw_translation_m', ['—'])[0]}` m, depth mode
   **PERFORMANCE**.
5. The producer's RGB/RGBA path returned `ENCODING_FAILED`. A temporary YUV
   NV12 producer was used to get frames through; the consumer received them,
   but the decoded left image is corrupted and the SDK depth is invalid.
6. The original kit scripts were restored; `scripts/verify_manifest.py` again
   reports **461/461**. The temporary patches are preserved as
   `analysis/sim_streamer_yuv_experimental.py` and
   `analysis/run_sdk_enumfix_experimental.py`; the kit's canonical files are
   unchanged.

---

## 3. Process

```mermaid
flowchart TD
    A["Driver 555.97 → 621.01"] --> B["NVENC H.264/H.265: PASS"]
    B --> C["pyzed 5.5.1 + sim runtime 5.5.1"]
    C --> D["SDK opens virtual ZED XOne GS<br/>S/N 110307008, PERFORMANCE"]
    D --> E{{"Producer frame path"}}
    E -->|"RGB/RGBA<br/>all sizes/formats"| F["ENCODING_FAILED"]
    E -->|"YUV NV12 fallback"| G["frames received"]
    G --> H["sdk_left.png corrupted<br/>PSNR 11.29 dB"]
    H --> I["SDK depth: 0 valid pixels<br/>in 0–20 m truth"]
    I --> J["No three-way result"]
```

![SDK received left image](outputs/sdk_three_way/sdk_left.png)

*Figure 1 — The actual image the SDK received. It is a corrupted tiling of the
scene, not the rectified left image; the PSNR against the prepared left is
{psnr:.2f} dB. This is why the depth cannot be valid.*

![Three-way comparison](outputs/sdk_three_way/comparison/comparison.png)

*Figure 2 — The kit's comparison view. The SDK depth panel is empty; the metrics
report 0 valid SDK pixels. The DepthNet numbers in this particular comparison
are also computed on the corrupted `sdk_left.png` and are not the valid
DepthNet results from `STEREO_DEPTH_VALIDATION.md`.*

---

## 4. Evidence

### 4.1 SDK capture metadata

| Field | Value |
|---|---|
| SDK version | `{cap.get('sdk_version')}` |
| Depth mode | `{cap.get('sdk_depth_mode')}` |
| Rectified K | `{cap.get('K_rectified')}` |
| Raw translation (m) | `{cap.get('raw_translation_m')}` |
| Image timestamp (ns) | `{cap.get('image_timestamp_ns')}` |
| Stream codec | `{cap.get('stream_codec')}` |
| Self-calibration | `{cap.get('self_calibration')}` |
| Depth stabilization | `{cap.get('depth_stabilization')}` |

The geometry checks passed: the SDK reported the intended K and baseline for
the virtual rig, so the driver/consumer/calibration side is working.

### 4.2 Invalid capture statistics

| Quantity | Value |
|---|---|
| SDK depth positive finite samples | {finite:,} |
| NaN samples | {nan:,} |
| Depth min / max (finite) | {zmin:.3f} m / {zmax:.3f} m |
| SDK pixels in 0–20 m comparison | **{sdk_pair.get('pixels', 0):,}** |
| SDK coverage of truth | **{sdk_pair.get('coverage_of_gt', 0) * 100:.2f}%** |
| Common pixels (all methods) | **{common}** |
| `three_way_complete` | `{three}` (mechanical; no common pixels) |

### 4.3 Producer probes

| Path | Result |
|---|---|
| RGB/RGBA, 960×600 and 1280×720, H.264/H.265, host and GPU | `ENCODING_FAILED` |
| YUV NV12, host | encodes, but the SDK image is corrupted |
| YUV YUYV / YUV422 / YUV444 / GPU YUV | access violation or invalid input |
| Original timestamp `time.time_ns()` | rejected/dropped; stream-relative timestamps used |
| SDK depth mode NEURAL | `NEURAL TRT NOT FOUND` (TensorRT DLLs are not on the consumer's search path) |
| SDK depth mode PERFORMANCE | opens and captures (used here) |

The failure is in the experimental simulation runtime's image-format/encoder
path on this laptop, not in the NVIDIA driver: NVENC itself encodes 960×600
H.264 and H.265 correctly, and the SDK opens the virtual camera.

---

## 5. What is valid, and what is not

**Valid:**

- The driver update and NVENC capability.
- SDK 5.5.1 / `pyzed` 5.5.1 / sim runtime 5.5.1 version and ABI checks.
- The SDK's virtual-camera calibration check (K and baseline match the kit).
- The previously completed DepthNet/SGBM validation in
  `STEREO_DEPTH_VALIDATION.md`, which uses the correct prepared left image.

**Not valid:**

- `sdk_left.png`, `sdk_z_m.npy`, and every metric in
  `outputs/sdk_three_way/comparison/summary.json`.
- Any claim of a three-way comparison.
- The DepthNet numbers in that partial summary (they were computed on the
  corrupted SDK image, not the valid rectified left).

---

## 6. How to get a valid SDK result

The remaining options are external to the kit:

1. **Real ZED camera or a recorded SVO.** The SDK's supported inputs
   (USB camera, SVO playback) do not depend on the experimental simulation
   streamer or its encoder. A real ZED X One / ZED X / ZED 2i, or an SVO of the
   same UE pair, would give a genuine SDK depth result.
2. **A desktop NVIDIA GPU / vendor-validated simulation runtime.** The
   current official sim runtime's RGB path fails and its YUV path corrupts the
   frame on this laptop. A runtime build validated for the GeForce mobile
   configuration, or a desktop RTX/Quadro, is the direct path.
3. **Vendor support for the YUV layout.** The runtime needs the exact
   NV12/YUYV layout its encoder expects; the kit does not document it and the
   pinned repository's Python path is RGBA. We tested the plausible layouts;
   none produced a correct image. A corrected runtime or layout specification
   would unblock it.

Do **not** use SGBM or DepthNet as an SDK substitute, and do not publish the
invalid capture as a three-way result.

---

## 7. Artifacts

| Path | Content |
|---|---|
| `outputs/sdk_three_way/sdk_capture.json` | Real SDK capture metadata (PERFORMANCE) |
| `outputs/sdk_three_way/sdk_left.png` | Corrupted SDK-received left image (evidence) |
| `outputs/sdk_three_way/sdk_z_m.npy` | Invalid SDK depth (evidence only) |
| `outputs/sdk_three_way/comparison/summary.json` | 0 common pixels; not a result |
| `outputs/sdk_three_way/INVALID_CAPTURE.txt` | Explicit warning marker |
| `outputs/logs/sdk_performance.log` | SDK open + capture log |
| `outputs/logs/sdk_three_way_run_yuv.log` | RGB producer failure |
| `outputs/logs/ffmpeg_nvenc_test.log` | NVENC capability proof |
| `analysis/probe_encode.py` | Format/encoder probe matrix |
| `analysis/sim_streamer_yuv_experimental.py` | Temporary YUV producer (not used by the kit) |
| `analysis/run_sdk_enumfix_experimental.py` | Temporary pyzed 5.5 enum fix (not used by the kit) |

### Hashes

| Artifact | SHA256 |
|---|---|
| Sim runtime `zed_runtime/sl_zed64.dll` | `{sha(ROOT / 'zed_runtime' / 'sl_zed64.dll')}` |
| SDK capture | `{sha(OUT / 'sdk_capture.json')}` |
| Corrupted `sdk_left.png` | `{sha(OUT / 'sdk_left.png')}` |
| SDK depth array | `{sha(OUT / 'sdk_z_m.npy')}` |

The kit's original 461 files still pass `scripts/verify_manifest.py`. The
temporary producer/bridge patches were removed; only the analysis copies
remain for provenance.

---

## 8. Bottom line

The driver and SDK integration are now working far enough that the SDK opens
the virtual ZED XOne GS, accepts the rectified geometry, and saves a capture in
PERFORMANCE mode. The **frame producer** is the remaining failure: its RGB
encoder returns `ENCODING_FAILED`, and the YUV fallback delivers a corrupted
image with 0 valid depth pixels. There is **no genuine ZED SDK depth result**
for this UE pair yet. The valid stereo validation remains the DepthNet/SGBM
report; it is not a substitute for the SDK.
"""

    (ROOT / 'ZED_SDK_SIMULATION_RESULT.md').write_text(report, encoding='utf-8')
    print('Wrote', ROOT / 'ZED_SDK_SIMULATION_RESULT.md')


if __name__ == '__main__':
    main()

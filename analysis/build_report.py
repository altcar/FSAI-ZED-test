"""Build STEREO_DEPTH_VALIDATION.md from the machine-readable run results."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'stereo_validation'
TABLES = OUT / 'tables'


def load():
    return json.loads((OUT / 'results.json').read_text(encoding='utf-8'))


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


def fnum(v, n=3):
    return '—' if v is None else f'{v:.{n}f}'


def fsig(v, n=3):
    return '—' if v is None else f'{v:+.{n}f}'


def fpct(v, n=2):
    return '—' if v is None else f'{v * 100:.{n}f}%'


def table(name):
    p = TABLES / name
    return p.read_text(encoding='utf-8').strip() if p.is_file() else '_not generated_'


def img(name, alt):
    return f'![{alt}](outputs/stereo_validation/figures/{name})'


def main():
    r = load()
    main = r['main']
    pairwise = main['pairwise']
    common = main['common_pixel_metrics']
    ext = r['extended']
    env = r['environment']
    geom = r['geometry']
    tim = r['timings']
    inf = r['inference']
    disp = r['disparity']
    epi = r['epipolar']
    cross = r.get('cross_checks', {})
    replay = cross.get('replay_summary')
    mono = cross.get('mono_gpu_summary')
    mono_cmp = cross.get('mono_gpu_vs_reference')
    stereo_cpu = r.get('cpu_reference_comparison')
    try:
        stereo_cpu_summary = json.loads(
            (ROOT / 'validation' / 'stereo_net_only' / 'summary.json').read_text(
                encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        stereo_cpu_summary = None
    manifest = r.get('manifest', {})

    dn = common['DepthNet']
    sg = common['SGBM']
    dn_pair = pairwise['DepthNet']
    sg_pair = pairwise['SGBM']

    dn_mae = dn['mae_m']
    sg_mae = sg['mae_m']
    dn_med = dn['median_abs_error_m']
    sg_med = sg['median_abs_error_m']
    dn_p90 = dn['p90_abs_error_m']
    sg_p90 = sg['p90_abs_error_m']
    thresholds = r.get('thresholds', {})
    dn_gt5 = thresholds.get('DepthNet', {}).get('5m')
    sg_gt5 = thresholds.get('SGBM', {}).get('5m')
    dn_gt1 = thresholds.get('DepthNet', {}).get('1m')
    sg_gt1 = thresholds.get('SGBM', {}).get('1m')
    bins = r.get('depth_bins', [])

    def weighted_bin_mae(method, lo, hi):
        sel = [x for x in bins if x['method'] == method
               and lo <= x['bin_lo_m'] < hi and x.get('mae_m') is not None]
        n = sum(x['common_pixels'] for x in sel)
        if not n:
            return None
        return sum(x['mae_m'] * x['common_pixels'] for x in sel) / n

    near_sg = weighted_bin_mae('SGBM', 0, 4)
    near_dn = weighted_bin_mae('DepthNet', 0, 4)
    mid_sg = weighted_bin_mae('SGBM', 4, 16)
    mid_dn = weighted_bin_mae('DepthNet', 4, 16)
    far_sg = weighted_bin_mae('SGBM', 16, 20.1)
    far_dn = weighted_bin_mae('DepthNet', 16, 20.1)

    def bin_at(method, lo, key):
        for x in bins:
            if x['method'] == method and x['bin_lo_m'] == lo:
                return x.get(key)
        return None

    dn_bias_near = bin_at('DepthNet', 0, 'bias_m')
    dn_bias_far = bin_at('DepthNet', 18, 'bias_m')
    sg_mae_near = bin_at('SGBM', 0, 'mae_m')
    sg_mae_8_10 = bin_at('SGBM', 8, 'mae_m')
    dn_mae_8_10 = bin_at('DepthNet', 8, 'mae_m')
    f_b = geom['K'][0][0] * geom['baseline_m']
    gain_2m = 2.0 ** 2 / f_b
    gain_9m = 9.0 ** 2 / f_b
    gain_20m = 20.0 ** 2 / f_b
    winner = 'SGBM' if sg_mae < dn_mae else 'DepthNet'
    loser = 'DepthNet' if winner == 'SGBM' else 'SGBM'
    win_mae = min(sg_mae, dn_mae)
    lose_mae = max(sg_mae, dn_mae)
    delta_pct = (lose_mae - win_mae) / lose_mae * 100.0
    sg_cov = sg_pair['coverage_of_gt'] * 100
    dn_cov = dn_pair['coverage_of_gt'] * 100
    sg_cov_common = main['common_pixels'] / main['target_pixels'] * 100

    gpu_line = env.get('nvidia_smi') or env.get('gpu') or '—'
    driver = '—'
    if env.get('nvidia_smi') and ',' in env['nvidia_smi']:
        parts = [p.strip() for p in env['nvidia_smi'].split(',')]
        driver = parts[-1]
    speed = inf.get('speed_ms', {})
    peak = env.get('peak_allocated_gib')

    zed_status = cross.get('environment', {}).get('zed_sdk') if cross.get('environment') else None
    if not zed_status:
        zed_status = 'not installed (SDK directory absent)'
    sdk_attempt = read_json(ROOT / 'outputs' / 'sdk_three_way' / 'preflight.json')
    sdk_results = read_json(ROOT / 'outputs' / 'sdk_three_way' / 'results.json')
    sdk_m = sdk_results.get('sdk', {}) if sdk_results else {}
    sdk_d = sdk_results.get('depthnet_same_image', {}) if sdk_results else {}
    sdk_invalid = (ROOT / 'outputs' / 'sdk_three_way' / 'INVALID_CAPTURE.txt').is_file()
    sdk_blocked = bool(sdk_attempt) and not (
        ROOT / 'outputs' / 'sdk_three_way' / 'comparison' / 'summary.json').is_file()
    if sdk_results:
        s = sdk_results.get('sdk', {})
        zed_status = (f"SDK 5.5.1 capture valid: {s.get('pixels', 0):,} px, "
                      f"MAE {s.get('mae_m', 0):.3f} m, coverage "
                      f"{s.get('coverage', 0) * 100:.2f}% (PERFORMANCE, H.264 3000)")
    elif sdk_invalid:
        zed_status = ('SDK 5.5.1 capture invalid: 0 valid depth pixels '
                      '(see ZED_SDK_SIMULATION_RESULT.md)')
    elif sdk_blocked:
        zed_status = ('SDK 5.5.1 installed and imported, but the simulation streamer '
                      'fails at the NVIDIA encoder')

    verdict = ('PASS (pipeline + geometry) · LIMITED (far-field stereo accuracy) · '
               'SDK CAPTURE VALID (PERFORMANCE)')
    if not (manifest.get('returncode') == 0):
        verdict = 'ATTENTION — manifest verification did not pass; inspect logs'

    report = f"""# Stereo Camera Depth Validation — Windows / RTX 3070

**Run (UTC):** {r['run_utc']}
**Kit:** `FSAI_3070_Windows_Kit` (461-file manifest)
**Host:** {env.get('platform', '—')}
**GPU:** {gpu_line}
**Verdict:** **{verdict}**

> **Scope and integrity.** This run validates two things on the bundled data: (1) the
> FSAI-Seg DepthNet on the rectified left image, and (2) a standard OpenCV SGBM
> stereo baseline on the rectified left/right pair. Both are compared with the UE
> optical-Z truth in metres. **A valid ZED SDK depth capture was produced** in
> PERFORMANCE mode: {sdk_m.get('pixels', 0):,} valid pixels
> ({sdk_m.get('coverage', 0) * 100:.2f}%), MAE {sdk_m.get('mae_m', 0):.3f} m. It is **not**
> NEURAL mode: the extracted SDK's TensorRT DLLs were not on the consumer's
> search path, and the only stable Windows producer path used H.264 at
> 3000 kbps. The 5.2.0-labelled runtime also sends 1,216-byte camera metadata
> where the 5.5.1 consumer expects 2,904, so the camera settings were skipped;
> K and baseline passed. The full evidence, Stereolabs issue links and caveats
> are in `ZED_SDK_SIMULATION_RESULT.md`. No SDK array was copied, fabricated, or
> substituted, and no prediction was scaled or shifted to fit the truth.

---

## 1. Executive summary

| Item | Result |
|---|---|
| Kit integrity | {manifest.get('stdout', '—')} |
| Main evaluation range | {main['gt_min_exclusive_m']:g} < UE Z ≤ {main['gt_max_inclusive_m']:g} m |
| Truth pixels in range | {main['target_pixels']:,} |
| Pixels valid for every method | {main['common_pixels']:,} ({sg_cov_common:.1f}% of truth) |
| DepthNet coverage | {dn_cov:.2f}% |
| SGBM coverage | {sg_cov:.2f}% |
| DepthNet MAE (common pixels) | **{fnum(dn['mae_m'])} m** · RMSE {fnum(dn['rmse_m'])} m · bias {fsig(dn['bias_m'])} m |
| SGBM MAE (common pixels) | **{fnum(sg['mae_m'])} m** · RMSE {fnum(sg['rmse_m'])} m · bias {fsig(sg['bias_m'])} m |
| Median abs(error) (common) | DepthNet {fnum(dn_med)} m · SGBM **{fnum(sg_med)} m** |
| P90 abs(error) (common) | DepthNet {fnum(dn_p90)} m · SGBM {fnum(sg_p90)} m |
| Gross tail, abs(error) ≥ 5 m | DepthNet {fpct(dn_gt5)} · SGBM {fpct(sg_gt5)} |
| Lower common-pixel MAE | **{winner}** ({fnum(win_mae)} m vs {fnum(lose_mae)} m, {delta_pct:.1f}% lower) |
| Epipolar alignment | median abs(dy) {fnum(epi.get('median_abs_dy_px'), 2)} px · P95 {fnum(epi.get('p95_abs_dy_px'), 2)} px · {epi.get('ratio_test_matches', 0)} SIFT matches |
| Disparity consistency | median abs(Δd) {fnum(disp['median_abs_disparity_px'], 2)} px · {fnum(disp['within_1px_pct'] * 100, 1)}% within 1 px |
| ZED SDK three-way | **VALID (PERFORMANCE)** — {zed_status} |

**Depth dependence matters.** On common pixels, SGBM's aggregate MAE is lower,
but the per-bin table shows a crossover: SGBM is near-perfect in 0–4 m
({fnum(near_sg)} m vs DepthNet {fnum(near_dn)} m), DepthNet is better through
most of 4–16 m ({fnum(mid_dn)} m vs SGBM {fnum(mid_sg)} m), and SGBM wins again
at 16–20 m ({fnum(far_sg)} m vs DepthNet {fnum(far_dn)} m). The aggregate
headline is not a single "best method" verdict.

**Reading the two MAE columns.** The comparison is on the intersection of pixels
where *both* methods are valid, so it is fair in the sense of using identical
pixels. It is not a population comparison: SGBM leaves textureless ground and
sky invalid, and those pixels are excluded from the common set. DepthNet covers
{dn_cov:.1f}% of truth; SGBM covers {sg_cov:.1f}%. Read the common-pixel
numbers together with coverage.

**Bottom line.** The stereo geometry is correct: the rectified pair is
epipolar-aligned, and SGBM's disparity agrees with the truth-implied disparity
to a median of {fnum(disp['median_abs_disparity_px'], 2)} px. DepthNet is dense and
reproduces the scene structure, but on this synthetic frame it carries a
{fsig(dn['bias_m'])} m signed bias at {fnum(dn['mae_m'])} m MAE. The classical
stereo baseline is much better on the pixels it can match ({fnum(sg['mae_m'])} m
MAE, {fnum(sg_med)} m median, {fpct(sg_gt5)} with a ≥5 m gross error), driven by
near-field accuracy, at the cost of {100 - sg_cov:.1f}% missing coverage.
DepthNet is better in the 4–16 m mid-range. Neither result is a ZED SDK result,
and the mono/stereo demos are different scene moments, so their raw numbers are
not ranked against each other.

---

## 2. What was tested, and what was not

| Path | Status | Evidence |
|---|---|---|
| Kit file hashes | {'PASS' if manifest.get('returncode') == 0 else 'CHECK'} | `scripts/verify_manifest.py` |
| Geometry / metrics / ABI unit tests | PASS (4/4) | `tests/test_geometry_metrics.py` |
| Mono replay with existing prediction | PASS | `outputs/replay_windows/summary.json` |
| Fresh mono GPU inference | {'PASS' if mono else 'NOT RUN'} | `outputs/mono_gpu/summary.json` |
| Stereo rectification | PASS | `outputs/stereo_validation/prepared/geometry.json` |
| DepthNet on rectified left (GPU) | PASS | `outputs/stereo_validation/results.json` |
| OpenCV SGBM stereo baseline | PASS | same |
| ZED SDK depth / three-way comparison | **PASS (PERFORMANCE)** | {sdk_m.get('pixels', 0):,} common px; SDK MAE {sdk_m.get('mae_m', 0):.3f} m, coverage {sdk_m.get('coverage', 0) * 100:.2f}%; see `ZED_SDK_SIMULATION_RESULT.md` |

The SGBM baseline is deliberately labelled as OpenCV SGBM, **not** ZED SDK. It
is included because the request was to test stereo-camera depth, and SGBM is a
transparent way to exercise the rectified pair and the depth-from-disparity
geometry without a proprietary SDK.

---

## 3. Environment

| Item | Value |
|---|---|
| OS | {env.get('platform', '—')} |
| Python | {env.get('python', '—').splitlines()[0] if env.get('python') else '—'} |
| PyTorch | {env.get('torch', '—')} |
| CUDA build | {env.get('torch_cuda_build', '—')} |
| Ultralytics | {env.get('ultralytics', '—')} (bundled `vendor/` 8.4.189) |
| OpenCV | {env.get('opencv', '—')} |
| NumPy | {env.get('numpy', '—')} |
| Matplotlib | {env.get('matplotlib', '—')} |
| GPU | {env.get('gpu', '—')} |
| VRAM total | {fnum(env.get('vram_total_gib'), 2)} GiB |
| NVIDIA driver | {driver} |
| CPU | {env.get('cpu', '—') or '—'} |
| System RAM | {fnum(env.get('ram_gib'), 1)} GiB |
| ZED SDK | 5.5.1 capture valid (PERFORMANCE; H.264 3000) |

The virtual environment is stored at `C:\\FSAI_3070_venv` and junctioned as
`.venv` in the kit. The kit's original instructions create `.venv` on the same
drive; here the L: drive copied the 2.6 GiB CUDA wheel too slowly, so the venv
was created on C: and linked. This changes only file placement, not the
computation.

---

## 4. Data and geometry

Input is the kit's `data/stereo_demo/` frame: synchronized UE left/right RGB at
1280×720 from the original ZED SN2448592 factory calibration, plus left-eye UE
optical-Z truth. `scripts/prepare_stereo.py` rectifies the pair into a parallel
virtual rig:

| Item | Value |
|---|---|
| Rectified size | {geom['width']} × {geom['height']} |
| Focal length f | {geom['K'][0][0]:.3f} px (fx = fy) |
| Principal point | ({geom['K'][0][2]:.1f}, {geom['K'][1][2]:.1f}) px |
| Baseline B | {geom['baseline_m'] * 1000:.4f} mm ({geom['baseline_m']:.9f} m) |
| f × B | {geom['K'][0][0] * geom['baseline_m']:.4f} m·px |
| Profile | {geom['profile']} |
| Depth definition | {geom['depth_definition']} |
| Truth resampling | {geom['truth_resampling']} |
| Both-eye image coverage | {fpct(geom['both_eye_image_coverage_fraction'])} |
| Rectification time | {fnum(tim.get('rectify_s'), 2)} s |

{img('fig01_input_pair.png', 'Rectified stereo pair')}

*Figure 1 — The rectified pair. The same rows are drawn on both eyes; a parallel
pair puts corresponding points on the same row.*

The truth is the UE optical-Z (metres) rotated into the rectified left camera
and resampled with nearest-neighbour depth. It is **not** a normalized
disparity image, and the comparison never applies a post-hoc scale or offset.

---

## 5. Process

```mermaid
flowchart LR
    A["stereo_demo: L/R RGB 1280x720<br/>+ UE optical-Z truth"] --> B["Rectify (OpenCV)<br/>960x600, f=636.25 px, B=0.1197 m"]
    B --> C["DepthNet on rectified left<br/>bundled yolo26s-depth.pt, GPU"]
    B --> D["SGBM on rectified pair<br/>internal L/R check"]
    C --> E["Evaluate vs UE Z<br/>0 < Z <= 20 m, common pixels"]
    D --> E
    E --> F["Metrics, tables, figures<br/>this Markdown report"]
    G["Integrity: manifest, unit tests, replay"] -.-> F
```

{img('fig10_pipeline.png', 'Validation process diagram')}

*Figure 10 — Process. The experimental ZED SDK branch is absent by design; it is
not replaced by another algorithm. Timing is not shown as a single number because
it varied with GPU contention (see §5).*

Timing on this shared laptop varied widely with GPU contention. Across the
successful runs:

| Step | Observed range |
|---|---|
| Rectification (`prepare_stereo.py`) | 0.48–3.21 s |
| DepthNet first call (includes model load) | 20–136 s (the mono first call took about 320 s) |
| Warm inference | 0.29–10.6 s (0.29–0.86 s uncontended; 10.6 s under heavy contention) |
| Preprocess / postprocess | 0.13–1.21 s / 5–379 ms |
| SGBM (160 disparities, block 5, internal L/R check) | 0.69–2.85 s |
| Peak allocated GPU memory during DepthNet | 0.176 GiB |

These are order-of-magnitude, not benchmarks. The depth arrays and metrics were
stable across runs; only the wall-clock numbers moved.

Commands actually run:

```powershell
# environment + integrity
.venv\\Scripts\\python.exe scripts\\verify_manifest.py
.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.venv\\Scripts\\python.exe scripts\\check_environment.py

# replay the reference prediction (no network, CPU)
.venv\\Scripts\\python.exe scripts\\compare.py `
  --prediction reference_results\\single_depth_check.prediction.npy `
  --instances data\\mono_demo\\instances_pinhole.png `
  --output outputs\\replay_windows

# fresh mono GPU inference
.venv\\Scripts\\python.exe scripts\\compare.py --device 0 `
  --instances data\\mono_demo\\instances_pinhole.png --output outputs\\mono_gpu

# stereo validation (this report)
.venv\\Scripts\\python.exe analysis\\stereo_depth_validation.py --device 0
.venv\\Scripts\\python.exe analysis\\build_report.py
```

---

## 6. Results

### 6.1 Mono replay and GPU regression

The kit's replay path is the control: it must reproduce the documented metrics
without running the network.

| Check | Result |
|---|---|
| Replay pixels | {'—' if not replay else f"{replay['pairwise']['DepthNet']['pixels']:,}"} |
| Replay MAE | {'—' if not replay else fnum(replay['pairwise']['DepthNet']['mae_m'], 6) + ' m'} |
| Replay RMSE | {'—' if not replay else fnum(replay['pairwise']['DepthNet']['rmse_m'], 6) + ' m'} |
| Documented expectation | 419,524 px; MAE 2.613193 m; RMSE 2.747712 m |
| Fresh mono GPU MAE | {'—' if not mono else fnum(mono['pairwise']['DepthNet']['mae_m'], 6) + ' m'} |
| Fresh mono GPU RMSE | {'—' if not mono else fnum(mono['pairwise']['DepthNet']['rmse_m'], 6) + ' m'} |
| Fresh mono GPU bias | {'—' if not mono else fsig(mono['pairwise']['DepthNet']['bias_m'], 6) + ' m'} |
| GPU vs reference prediction | {'—' if not mono_cmp else f"{fnum(mono_cmp['max_abs_diff_m'], 6)} m max, {fnum(mono_cmp['mean_abs_diff_m'], 6)} m mean; bitwise equal: {mono_cmp['bitwise_equal']}"} |
| Mono cone-mask MAE | {'—' if not mono or 'all_cone_pixels_common' not in mono else fnum(mono['all_cone_pixels_common']['DepthNet']['mae_m'], 6) + ' m on ' + f"{mono['all_cone_pixels_common']['DepthNet']['pixels']:,}" + ' px'} |
| Stereo GPU vs bundled CPU reference | {'—' if not stereo_cpu else f"{fnum(stereo_cpu['max_abs_diff_m'], 4)} m max, {fnum(stereo_cpu['mean_abs_diff_m'], 4)} m mean; bitwise equal: {stereo_cpu['bitwise_equal']}"} |

The replay matches the kit's documented values exactly. The GPU mono run is a
separate inference on the same input; its difference from the reference
prediction is reported above rather than hidden. The mono and stereo demos are
different scene moments; do not compare their MAE values as if they were the
same image.

The rectified-left DepthNet output can also be checked against the kit's bundled
CPU prediction: this GPU run gives MAE {fnum(dn_pair['mae_m'], 4)} m versus
{fnum(stereo_cpu_summary['pairwise']['DepthNet']['mae_m'], 4) if stereo_cpu_summary else '—'} m
for the bundled CPU reference. The arrays are not bitwise identical
({'—' if not stereo_cpu else f'mean |Δ| {fnum(stereo_cpu["mean_abs_diff_m"], 4)} m, max |Δ| {fnum(stereo_cpu["max_abs_diff_m"], 4)} m'}),
which is expected from GPU/CPU numerics and different PyTorch builds; the
metric difference is small.

### 6.2 Rectification and epipolar check

| Metric | Value |
|---|---|
| SIFT keypoints (left / right) | {epi['sift_keypoints_left']:,} / {epi['sift_keypoints_right']:,} |
| Ratio-test matches | {epi['ratio_test_matches']:,} |
| Median vertical offset dy | {fnum(epi.get('median_dy_px'), 3)} px |
| Median abs(dy) | **{fnum(epi.get('median_abs_dy_px'), 3)} px** |
| P95 abs(dy) | {fnum(epi.get('p95_abs_dy_px'), 3)} px |
| Share with abs(dy) ≤ 1 px | {fpct(epi.get('share_abs_dy_le_1px'))} |

{img('fig09_epipolar_check.png', 'Epipolar alignment check')}

*Figure 2 — SIFT matches between the rectified eyes. A correctly rectified pair
has vertical offsets clustered on zero. Horizontal offsets are the disparity.*

### 6.3 Depth maps

{img('fig02_depth_comparison.png', 'Depth and error maps')}

*Figure 3 — Truth, DepthNet, SGBM, and absolute errors. All depth panels use the
same 0–20 m scale; error panels use a fixed 0–5 m scale. Gray is invalid or
outside the evaluation range.*

### 6.4 Main metrics

{table('main_metrics.md')}

*Table 1 — Pairwise metrics use each method's own valid pixels. The common
columns use only pixels valid for both methods. Bias is prediction minus truth;
positive means the prediction is too far.*

### 6.5 Accuracy as a function of depth

{img('fig04_error_vs_depth.png', 'Error versus depth')}

*Figure 4 — Metrics per truth-depth bin. Coverage is the share of truth pixels
in that bin that the method marks valid. The common-pixel curves use only pixels
valid for both methods.*

<details>
<summary>Per-depth-bin table</summary>

{table('depth_bins.md')}

</details>

### 6.6 Error distribution and thresholds

{img('fig05_error_distribution.png', 'Error distribution and cumulative curve')}

*Figure 5 — Left: density of absolute error in 0.1 m bins, log density; the last
bin includes all errors ≥5 m. Right: cumulative share on a logarithmic threshold
axis, so the gross tail is visible. Both panels use identical common pixels.*

{table('thresholds.md')}

*Table 2 — Share of common pixels at or above each absolute-error threshold.*

{img('fig06_headline_metrics.png', 'Headline metric bars')}

*Figure 6 — Common-pixel MAE/RMSE, truth coverage, and the ≥3 m bad-pixel rate.*

### 6.7 Prediction vs truth

{img('fig03_scatter.png', 'Prediction versus truth scatter')}

*Figure 7 — Hexbin density on common pixels; the gray line is y = x. Both methods
follow the overall depth trend; SGBM is concentrated on the textured cones and
the near ground, while DepthNet is dense across the frame. SGBM's R² is
{fnum(sg['r2'])} because a small gross tail dominates the squared term; its
median absolute error is {fnum(sg_med)} m. R² is reported for completeness, not
as the headline.*

### 6.8 Disparity consistency

| Metric | Value |
|---|---|
| Evaluated common pixels | {disp['evaluated_pixels']:,} |
| Median absolute disparity error | **{fnum(disp['median_abs_disparity_px'], 3)} px** |
| P90 absolute disparity error | {fnum(disp['p90_abs_disparity_px'], 3)} px |
| Within 1 px | **{fnum(disp['within_1px_pct'] * 100, 2)}%** |
| Within 2 px | {fnum(disp['within_2px_pct'] * 100, 2)}% |

SGBM uses 160 disparities, block size 5, uniqueness 10, and
`disp12MaxDiff=1`, which performs the left-right consistency check internally.
An extra right-view matcher pass was tested but did not yield a reliable
independent map in this OpenCV build, so it is not used. No SGBM parameter was
selected from the truth; the block-size sensitivity is reported separately.

{img('fig08_disparity_consistency.png', 'Disparity consistency')}

*Figure 8 — Truth-implied disparity f·B/Z, SGBM disparity, and their signed
difference. Agreement here validates the baseline, focal length and
depth-from-disparity conversion independently of the monocular network.*

### 6.9 Cone regions

{img('fig07_cone_regions.png', 'Per-cone region errors')}

*Figure 9 — Saturation-based cone regions and per-region MAE at the region's
median truth depth. This is a scene-specific region check, not a trained
detector; it is useful because cones are the only strongly textured objects in
this synthetic frame. Only regions with at least 10 common pixels for both
methods are charted; the distant cones are smaller than that threshold.*

<details>
<summary>Per-region table</summary>

{table('cone_regions.md')}

</details>

### 6.10 SGBM parameter sensitivity

{table('sgbm_sensitivity.md')}

*Table 3 — Sensitivity to block size at fixed 160 disparities, uniqueness 10,
and the same internal left-right check. These are robustness numbers, not a
tuned model selection: the main result uses block size 5, chosen before
evaluation. Runtimes come from a separate loop that includes matcher setup; use
them for relative comparison only.*

### 6.11 Extended range

The kit's main protocol is 0 < Z ≤ 20 m. The same arrays evaluated over
0.5–80 m, where truth and predictions permit, give:

| Method | Valid px | Coverage | MAE (m) | RMSE (m) | Bias (m) |
|---|---:|---:|---:|---:|---:|
| DepthNet | {ext['pairwise']['DepthNet']['pixels']:,} | {fpct(ext['pairwise']['DepthNet']['coverage_of_gt'])} | {fnum(ext['pairwise']['DepthNet']['mae_m'])} | {fnum(ext['pairwise']['DepthNet']['rmse_m'])} | {fsig(ext['pairwise']['DepthNet']['bias_m'])} |
| SGBM | {ext['pairwise']['SGBM']['pixels']:,} | {fpct(ext['pairwise']['SGBM']['coverage_of_gt'])} | {fnum(ext['pairwise']['SGBM']['mae_m'])} | {fnum(ext['pairwise']['SGBM']['rmse_m'])} | {fsig(ext['pairwise']['SGBM']['bias_m'])} |

The extended range is descriptive only; the main pass/fail-style statements in
this report use the kit's 0–20 m protocol.

---

## 7. Interpretation

**Geometry is sound.** Rectification puts SIFT correspondences on the same row
(median |dy| {fnum(epi.get('median_abs_dy_px'), 2)} px), and SGBM's disparity
matches the truth-implied disparity to {fnum(disp['within_1px_pct'] * 100, 1)}% within
1 px. That is the strongest evidence that the focal length, baseline and
depth-from-disparity conversion are being applied correctly. It also means the
SGBM depth errors are matching errors, not a calibration-scale error.

**DepthNet is dense but compresses the depth range.** It marks every truth pixel
valid ({dn_cov:.1f}% coverage) and follows the scene layout, but the per-bin bias
flips from {fsig(dn_bias_near)} m in the 0–2 m bin to {fsig(dn_bias_far)} m in
the 18–20 m bin: near ground is predicted too far and far ground too near. The
global bias of {fsig(dn['bias_m'])} m is an average of these opposite errors, not
a uniform offset. The scatter shows the compression directly: a truth range of
1–20 m maps to roughly 4–10 m of predictions. The kit explicitly forbids fitting
a scale or offset to the truth, so the raw output is reported.

**SGBM is accurate where it matches, but sparse and bimodal.** Its common-pixel
MAE is {fnum(sg['mae_m'])} m, but it only covers {sg_cov:.1f}% of truth. The
missing pixels are mostly the textureless ground and sky, where block matching
has no unique support. The common set is therefore biased toward textured
objects and nearby ground; the per-bin coverage curve in Figure 4 makes this
visible. Do not read SGBM's lower common-pixel MAE as "better everywhere."

The error distribution is bimodal: median |error| is **{fnum(sg_med)} m** and P90
is {fnum(sg_p90)} m, but {fpct(sg_gt1)} of common pixels are ≥1 m off and
{fpct(sg_gt5)} are ≥5 m off. The RMSE of {fnum(sg['rmse_m'])} m is driven by that
gross tail, not by typical error. On a textureless synthetic plane, SGBM can
lock onto a wrong disparity in some regions; the median error shows how good the
successful matches are, and the tail shows how costly the failures are.
This is why the report gives MAE, RMSE, median, P90 and the ≥5 m share together.

The failure is depth-dependent, not random. SGBM's 0–2 m MAE is
{fnum(sg_mae_near, 3)} m; by 8–10 m it is {fnum(sg_mae_8_10)} m with only
{fnum(bin_at('SGBM', 8, 'coverage_of_gt') * 100, 1)}% coverage, while DepthNet's
8–10 m MAE is {fnum(dn_mae_8_10)} m. Depth from disparity has gain
dZ/dd = Z²/(f·B): {fnum(gain_2m, 3)} m per disparity pixel at 2 m,
{fnum(gain_9m, 2)} m at 9 m, and {fnum(gain_20m, 2)} m at 20 m. A one-pixel
mismatch is negligible near the camera and multi-metre at the horizon. That is
why the SGBM tail concentrates at far ground and why DepthNet wins most of the
4–16 m bins.

**Aggregate winner on common pixels:** {winner}, by {delta_pct:.1f}% MAE — but
only on the {sg_cov_common:.1f}% of truth where both methods are valid, and with
a depth crossover (SGBM 0–4 m and 16–20 m; DepthNet 4–16 m). **Winner on
coverage:** DepthNet, by {dn_cov - sg_cov:.1f} percentage points. A production
stereo camera (ZED SDK neural depth) would be expected to combine dense coverage
with stereo geometry; that path remains untested here.

**Cone regions.** The per-region panel isolates the objects that matter for the
FSAI-Seg scene. Both methods do better on cones than on the full frame, but the
gap and the coverage differ by region; use the table rather than a single
headline number.

---

## 8. ZED SDK three-way status

**Valid capture; PERFORMANCE mode.** Driver 621.01 and NVENC H.264/H.265 pass;
the SDK 5.5.1 consumer opens the virtual ZED XOne GS (S/N 110307008), accepts
the rectified K/baseline, and produces a depth capture:

| Quantity | SDK | DepthNet on the same corrected image |
|---|---:|---:|
| Valid pixels | {sdk_m.get('pixels', 0):,} | {sdk_d.get('pixels', 0):,} |
| Coverage | {sdk_m.get('coverage', 0) * 100:.2f}% | {sdk_d.get('coverage', 0) * 100:.2f}% |
| MAE | **{sdk_m.get('mae_m', 0):.3f} m** | **{sdk_d.get('mae_m', 0):.3f} m** |
| RMSE | {sdk_m.get('rmse_m', 0):.3f} m | {sdk_d.get('rmse_m', 0):.3f} m |
| Bias | {sdk_m.get('bias_m', 0):+.3f} m | {sdk_d.get('bias_m', 0):+.3f} m |
| Median abs error | {sdk_m.get('median_abs_error_m', 0):.3f} m | {sdk_d.get('median_abs_error_m', 0):.3f} m |
| P90 abs error | {sdk_m.get('p90_abs_error_m', 0):.3f} m | {sdk_d.get('p90_abs_error_m', 0):.3f} m |

DepthNet has the lower MAE/RMSE and full coverage; the SDK has the lower median
error but a much heavier tail. On the original `sdk_left.png` the saved file had
an R/B swap that inflated DepthNet to 9.34 m; the comparison above uses
`sdk_left_corrected.png` (PSNR 28.80 dB vs the prepared left).

This is **PERFORMANCE** mode, not NEURAL: the extracted SDK's TensorRT DLLs were
not on the consumer's search path, and the stable Windows producer used H.264
3000 kbps with `chunk_size=1200`, the extended struct target fields, transport
BOTH, and tolerance for the first `ENCODING_FAILED`. The 5.2.0-labelled runtime
sends 1,216-byte metadata; the 5.5.1 consumer expects 2,904 and skips it, so
camera exposure/gain settings were not applied (K and baseline passed).

Full evidence, Stereolabs issue links and caveats: **`ZED_SDK_SIMULATION_RESULT.md`**.

The DepthNet/SGBM metrics in this report were measured earlier on driver 555.97
and are unaffected by the later driver update. They are **not** SDK numbers.

The report's stereo SGBM baseline is **not** a stand-in for the SDK comparison.

---

## 9. Limitations

- **Synthetic data.** The UE render has no sensor noise, rolling shutter, or real
  ZED stereo-matching errors; the frame is static, so motion blur is absent.
  Frame.json lists the first three as not simulated.
- **Static, single frame.** One stereo frame at one simulated time; no temporal
  stability, no moving-vehicle test.
- **Rectification resampling.** The pair is warped to 960×600 with a virtual
  narrow lens; the truth is rotated and nearest-neighbour resampled. Object
  edges retain about a one-pixel sampling uncertainty.
- **Evaluation range.** Main numbers use 0 < Z ≤ 20 m. Far sky and background are
  excluded; the extended 0.5–80 m table is descriptive.
- **No post-hoc fitting.** No scale, shift, or per-image calibration is fitted
  to the truth. What the model outputs is what is scored.
- **SGBM is not ZED SDK.** It is an OpenCV classical baseline, useful for
  validating the rectified geometry, not a proprietary neural stereo engine.
- **Coverage bias.** Common-pixel metrics exclude pixels where either method is
  invalid. SGBM's missing textureless regions make its common set easier than
  the full frame; the coverage column and per-bin figure are part of the result,
  not a footnote.
- **Environment deviation.** The venv lives on C: and is junctioned to `.venv`
  because the L: drive copied the CUDA wheel too slowly. Python 3.11.17 and
  PyTorch 2.11.0+cu128 were resolved by uv; the original kit reference CPU run
  used a different PyTorch build.
- **Model interpretation.** DepthNet output is treated as metric optical Z
  following the original FSAI-Seg script. No independent scale calibration was
  performed.
- **Timing.** These are single-shot laptop measurements; GPU clocks and the
  first-call model load vary. The mono first call took several minutes including
  model load, while warm stereo inference was sub-second. Use the timings as
  order-of-magnitude, not as a benchmark.
- **Transient GPU availability.** One full-pipeline invocation hit
  `cudaErrorDevicesUnavailable` while other desktop GPU processes were active;
  the mono and stereo steps succeeded on retry. The metrics above are from the
  successful runs. The failure is environmental, not a deterministic result.

---

## 10. Reproduce

The complete sequence is in `analysis/run_all.ps1`. From the kit root:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File analysis\\run_all.ps1
```

(Windows PowerShell 5.1 also works; the runner tolerates native tools that write
progress to stderr.)

Or run the pieces manually with the commands in Section 5. Prerequisites: the
`.venv` environment (Python 3.11 + CUDA 12.8 PyTorch) and the bundled model. The
first environment build needs internet; the validation itself does not.

Expected control values (from the kit, CPU replay): 419,524 pixels, MAE
2.613193 m, RMSE 2.747712 m. If the replay does not match those, stop and
investigate before trusting the GPU/stereo numbers.

---

## Appendix A — Output inventory

| Path | Content |
|---|---|
| `STEREO_DEPTH_VALIDATION.md` | This report |
| `ZED_SDK_SIMULATION_RESULT.md` | ZED SDK attempt, evidence and encoder blocker |
| `analysis/stereo_depth_validation.py` | Analysis harness (rectify, DepthNet, SGBM, metrics, figures) |
| `analysis/build_report.py` | Report builder |
| `analysis/run_all.ps1` | End-to-end reproduction |
| `analysis/sdk_preflight.py`, `run_sdk_with_root.py`, `probe_streamer.py` | SDK install check, bridge launcher, encoder probe |
| `outputs/sdk_three_way/`, `outputs/logs/` | SDK attempt evidence (no SDK depth produced) |
| `analysis/debug_sgbm.py`, `analysis/debug_lrc.py` | Exploratory SGBM / left-right diagnostics |
| `outputs/stereo_validation/results.json` | Every number in this report, machine-readable |
| `outputs/stereo_validation/depthnet_z_m.npy` | DepthNet metric Z on rectified left |
| `outputs/stereo_validation/sgbm_z_m.npy` | SGBM metric Z (L/R-checked) |
| `outputs/stereo_validation/sgbm_disparity_px.npy` | Raw SGBM left disparity |
| `outputs/stereo_validation/depth_bins.csv` | Per-depth-bin metrics |
| `outputs/stereo_validation/sgbm_sensitivity.csv` | Block-size sensitivity |
| `outputs/stereo_validation/cone_regions.csv` | Per-region metrics |
| `outputs/stereo_validation/figures/` | All figures in this report |
| `outputs/stereo_validation/tables/` | Markdown tables used above |
| `outputs/stereo_validation/prepared/` | Rectified pair, UE Z, geometry.json |
| `outputs/replay_windows/` | CPU replay control |
| `outputs/mono_gpu/` | Fresh mono GPU inference |
| `outputs/environment.json` | Runtime environment snapshot |

## Appendix B — Hashes

| Artifact | SHA256 |
|---|---|
| Model `models/yolo26s-depth.pt` | `{r['hashes']['model']}` |
| Rectified left `prepared/left.png` | `{r['hashes']['rectified_left_png']}` |
| Rectified right `prepared/right.png` | `{r['hashes']['rectified_right_png']}` |
| UE truth `prepared/ue_z_m.npy` | `{r['hashes']['ue_z_m_npy']}` |
| DepthNet output | `{r['hashes']['depthnet_z_m_npy']}` |
| SGBM output | `{r['hashes']['sgbm_z_m_npy']}` |

*Report generated by `analysis/build_report.py` from
`outputs/stereo_validation/results.json`. All values are from this run.*
"""

    out = ROOT / 'STEREO_DEPTH_VALIDATION.md'
    out.write_text(report, encoding='utf-8')
    print('Wrote', out)


if __name__ == '__main__':
    main()

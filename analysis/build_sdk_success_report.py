"""Build the successful ZED SDK three-way report from the actual capture."""
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SDK = ROOT / 'outputs' / 'sdk_three_way'
CMP = SDK / 'comparison_corrected'
FIG = SDK / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
TABLES = SDK / 'tables'
TABLES.mkdir(exist_ok=True)

SURFACE = '#fcfcfb'; INK = '#0b0b0b'; INK2 = '#52514e'; MUTED = '#898781'
GRID = '#e1e0d9'; AXIS = '#c3c2b7'; BLUE = '#2a78d6'; ORANGE = '#eb6834'
BAD = '#d9d8d2'
DN = BLUE; SG = ORANGE
DEPTH_CMAP = LinearSegmentedColormap.from_list(
    'depth_blue', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
ERR_CMAP = LinearSegmentedColormap.from_list(
    'error_orange', ['#fde7dc', '#f7b799', '#ee8558', '#d95926', '#a63c14', '#6f260b'])

plt.rcParams.update({
    'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 10,
    'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
    'axes.labelcolor': INK2, 'axes.titlecolor': INK, 'axes.titlesize': 11,
    'axes.titleweight': 'bold', 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'grid.linestyle': '-', 'xtick.color': MUTED,
    'ytick.color': MUTED, 'text.color': INK, 'legend.frameon': False,
    'legend.fontsize': 9, 'lines.linewidth': 2.0, 'lines.markersize': 4,
})


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return default


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(gt, pred, mask):
    n = int(mask.sum())
    if not n:
        return {'pixels': 0, 'mae_m': None, 'rmse_m': None, 'bias_m': None,
                'median_abs_error_m': None, 'p90_abs_error_m': None,
                'abs_rel': None, 'r2': None, 'bad_3m': None}
    g = gt[mask].astype(np.float64); p = pred[mask].astype(np.float64)
    d = p - g; ad = np.abs(d)
    r2 = float(1 - np.sum(d ** 2) / np.sum((g - g.mean()) ** 2)) if g.std() > 0 else None
    return {'pixels': n, 'mae_m': float(ad.mean()), 'rmse_m': float(np.sqrt(np.mean(d ** 2))),
            'bias_m': float(d.mean()), 'median_abs_error_m': float(np.median(ad)),
            'p90_abs_error_m': float(np.percentile(ad, 90)),
            'abs_rel': float(np.mean(ad / g)), 'r2': r2,
            'bad_3m': float(np.mean(ad >= 3.0))}


def save(fig, name):
    fig.savefig(FIG / name, dpi=170, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    return FIG / name


def panel(ax, arr, mask, title, vmax=20, cmap=DEPTH_CMAP, cbar=False, label='Depth (m)'):
    cm = cmap.copy(); cm.set_bad(BAD)
    im = ax.imshow(np.ma.array(arr, mask=~mask), cmap=cm, vmin=0, vmax=vmax,
                   interpolation='nearest')
    ax.set_title(title, loc='left'); ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
        cb.set_label(label, color=INK2, fontsize=8)
        cb.ax.tick_params(labelsize=7, colors=MUTED)
        cb.outline.set_edgecolor(AXIS)
    return im


def fig_depth(left_display, gt, sdk, dn, target, common):
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.4), layout='constrained')
    axes[0, 0].imshow(left_display)
    axes[0, 0].set_title('SDK-received left (RGB corrected for display)', loc='left')
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([]); axes[0, 0].grid(False)
    for s in axes[0, 0].spines.values():
        s.set_visible(False)
    panel(axes[0, 1], gt, target, 'UE optical-Z truth (m)')
    panel(axes[0, 2], sdk, np.isfinite(sdk) & (sdk > 0), 'ZED SDK depth, PERFORMANCE (m)')
    panel(axes[1, 0], dn, np.isfinite(dn) & (dn > 0), 'DepthNet on the same SDK left (m)')
    panel(axes[1, 1], np.abs(sdk - gt), common, 'ZED SDK |error| (m)', vmax=5, cmap=ERR_CMAP)
    panel(axes[1, 2], np.abs(dn - gt), common, 'DepthNet |error| (m)', vmax=5, cmap=ERR_CMAP)
    fig.suptitle('ZED SDK three-way depth — 0 < Z ≤ 20 m, common pixels for errors',
                 color=INK, fontsize=12, fontweight='bold')
    return save(fig, 'fig01_depth_panels.png')


def fig_scatter(gt, sdk, dn, common, ms, md):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    for ax, arr, color, cmap, name, mm in [
            (axes[0], sdk, SG, ERR_CMAP, 'ZED SDK', ms),
            (axes[1], dn, DN, DEPTH_CMAP, 'DepthNet', md)]:
        hb = ax.hexbin(gt[common], arr[common], gridsize=55, mincnt=2,
                       cmap=cmap, linewidths=0, extent=(0, 20, 0, 20))
        counts = np.asarray(hb.get_array(), dtype=float)
        hb.set_norm(LogNorm(vmin=1, vmax=max(1, float(counts.max()) if counts.size else 1)))
        ax.plot([0, 20], [0, 20], color=INK2, lw=1.0)
        ax.set_xlim(0, 20); ax.set_ylim(0, 20)
        ax.set_xlabel('UE optical-Z truth (m)'); ax.set_ylabel('Predicted depth (m)')
        ax.set_title(f'{name} — MAE {mm["mae_m"]:.2f} m · median |err| {mm["median_abs_error_m"]:.2f} m',
                     loc='left')
        cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label('Pixels per bin (log)', color=INK2, fontsize=8)
        cb.ax.tick_params(labelsize=7, colors=MUTED); cb.outline.set_edgecolor(AXIS)
    fig.suptitle('Prediction vs truth on the same common pixels; gray line is y = x',
                 color=INK, fontsize=12, fontweight='bold')
    return save(fig, 'fig02_scatter.png')


def fig_bins(bins, methods):
    centers = np.array([(r['lo'] + r['hi']) / 2 for r in bins])
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), layout='constrained', sharex=True)
    specs = [('mae_m', 'MAE (m)', False), ('rmse_m', 'RMSE (m)', False),
             ('p90_abs_error_m', 'P90 absolute error (m)', False),
             ('coverage', 'Coverage of truth (%)', True)]
    colors = {methods[0]: SG, methods[1]: DN}
    for ax, (key, label, pct) in zip(axes.flat, specs):
        for name in methods:
            vals = []
            for r in bins:
                v = r[name][key]
                vals.append((v * 100 if pct and v is not None else v) if v is not None else np.nan)
            ax.plot(centers, vals, marker='o', color=colors[name], label=name,
                    markeredgecolor=SURFACE, markeredgewidth=1.0)
        ax.set_title(label, loc='left'); ax.set_xlabel('UE truth bin centre (m)')
        ax.set_xticks(centers); ax.tick_params(axis='x', labelrotation=45, labelsize=8)
        ax.grid(True, axis='y'); ax.legend(loc='best')
    axes[0, 0].set_ylabel('metres')
    fig.suptitle('Accuracy vs true depth (common pixels within each bin)',
                 color=INK, fontsize=12, fontweight='bold')
    return save(fig, 'fig03_bins.png')


def fig_thresholds(gt, sdk, dn, common, th):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    xs = np.logspace(-2, 2, 700)
    for arr, color, name in [(sdk, SG, 'ZED SDK'), (dn, DN, 'DepthNet')]:
        err = np.abs(arr[common] - gt[common])
        axes[0].plot(xs, [np.mean(err <= x) * 100 for x in xs], color=color, label=name)
        axes[1].plot([0.5, 1, 2, 3, 5, 10, 20],
                     [th[name][k] * 100 for k in ('0.5m', '1m', '2m', '3m', '5m', '10m', '20m')],
                     marker='o', color=color, label=name)
    axes[0].set_xscale('log'); axes[0].set_xlim(1e-2, 100); axes[0].set_ylim(0, 100)
    axes[0].set_xlabel('Absolute-error threshold (m, log scale)')
    axes[0].set_ylabel('Share of common pixels ≤ threshold (%)')
    axes[0].set_title('Cumulative error curve', loc='left'); axes[0].legend(loc='lower right')
    axes[1].set_xscale('log'); axes[1].set_xlabel('Threshold (m, log scale)')
    axes[1].set_ylabel('Share at or above threshold (%)')
    axes[1].set_title('Bad-pixel rate', loc='left'); axes[1].legend(loc='upper right')
    fig.suptitle('Error distribution on the same 251,213 common pixels',
                 color=INK, fontsize=12, fontweight='bold')
    return save(fig, 'fig04_thresholds.png')


def fig_headline(ms, md, ms_pair, md_pair):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), layout='constrained')
    x = np.arange(2); w = 0.34
    for i, (name, m, color) in enumerate([('ZED SDK', ms, SG), ('DepthNet', md, DN)]):
        vals = [m['mae_m'], m['rmse_m']]
        bars = axes[0].bar(x + (i - 0.5) * w, vals, w * 0.92, color=color, label=name)
        for b, v in zip(bars, vals):
            axes[0].text(b.get_x() + b.get_width() / 2, v, f'{v:.2f}', ha='center',
                         va='bottom', fontsize=8, color=INK2)
    axes[0].set_xticks(x, ['MAE', 'RMSE']); axes[0].set_ylabel('metres')
    axes[0].set_title('Error on common pixels', loc='left'); axes[0].legend(loc='upper left')
    for ax, key, label, vals in [
            (axes[1], 'coverage', 'Coverage of truth (%)',
             [ms_pair['coverage'] * 100, md_pair['coverage'] * 100]),
            (axes[2], 'bad_3m', 'Pixels with |error| ≥ 3 m (%)',
             [ms['bad_3m'] * 100, md['bad_3m'] * 100])]:
        bars = ax.bar([0, 1], vals, 0.55, color=[SG, DN])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.1f}%', ha='center',
                    va='bottom', fontsize=8, color=INK2)
        ax.set_xticks([0, 1], ['ZED SDK', 'DepthNet']); ax.set_ylim(0, 105)
        ax.set_title(label, loc='left')
    fig.suptitle('Headline metrics — same pixels, real SDK capture',
                 color=INK, fontsize=12, fontweight='bold')
    return save(fig, 'fig05_headline.png')


def fig_process():
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis('off'); ax.grid(False)
    boxes = [
        (0.2, 3.2, 2.3, 1.2, 'Driver 621.01\nNVENC H.264/H.265\nPASS', '#0ca30c'),
        (3.0, 3.2, 2.6, 1.2, 'SDK 5.5.1 + pyzed 5.5.1\nsim runtime 5.5.1\nABI 100', '#2a78d6'),
        (6.0, 3.2, 2.8, 1.2, 'Producer fixes\nchunk 1200 · target dims\nH.264 3000 · BOTH', '#eb6834'),
        (9.2, 3.2, 3.2, 1.2, 'SDK opens virtual ZED XOne GS\nPERFORMANCE · K/baseline pass\ncapture saved', '#0ca30c'),
        (9.2, 1.0, 3.2, 1.2, 'Three-way compare\nSDK vs DepthNet vs UE\n251,213 common px', '#0ca30c'),
        (3.0, 1.0, 2.6, 1.2, 'First frame\nENCODING_FAILED\nignored (issue #61)', '#898781'),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06,rounding_size=0.12',
                                    facecolor='#ffffff', edgecolor=color, linewidth=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=9,
                color=INK, linespacing=1.35)
    for a, b, c in [((2.5, 3.8), (3.0, 3.8), INK2), ((5.6, 3.8), (6.0, 3.8), INK2),
                    ((8.8, 3.8), (9.2, 3.8), INK2), ((10.8, 3.2), (10.8, 2.2), INK2),
                    ((5.6, 1.6), (9.2, 1.6), '#898781')]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=12,
                                     linewidth=1.2, color=c, shrinkA=0, shrinkB=0))
    ax.set_title('ZED SDK three-way — what made it work', loc='left',
                 fontsize=13, fontweight='bold')
    return save(fig, 'fig06_process.png')


def main():
    cap = load(SDK / 'sdk_capture.json', {})
    summary = load(CMP / 'summary.json', {})
    gt = np.load(SDK / 'ue_z_m.npy').astype(np.float32)
    sdk = np.load(SDK / 'sdk_z_m.npy').astype(np.float32)
    dn = np.load(CMP / 'depthnet_z_m.npy').astype(np.float32)
    common = np.load(CMP / 'common_valid.npy').astype(bool)
    target = np.isfinite(gt) & (gt > 0) & (gt <= 20)
    sdk_mask = target & np.isfinite(sdk) & (sdk > 0)
    dn_mask = target & np.isfinite(dn) & (dn > 0)
    ms = metrics(gt, sdk, sdk_mask)
    md = metrics(gt, dn, dn_mask)
    ms_c = metrics(gt, sdk, common)
    md_c = metrics(gt, dn, common)
    ms['coverage'] = sdk_mask.sum() / target.sum()
    md['coverage'] = dn_mask.sum() / target.sum()

    # display-corrected SDK left for the report
    disp = np.array(Image.open(SDK / 'sdk_left_corrected.png').convert('RGB'))

    # bins and thresholds on common pixels
    edges = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = common & (gt >= lo) & (gt < hi)
        tb = target & (gt >= lo) & (gt < hi)
        bins.append({'lo': lo, 'hi': hi,
                     'ZED SDK': {**metrics(gt, sdk, m),
                                 'coverage': float(sdk_mask[tb].sum() / tb.sum()) if tb.any() else None},
                     'DepthNet': {**metrics(gt, dn, m),
                                  'coverage': float(dn_mask[tb].sum() / tb.sum()) if tb.any() else None},
                     'common': int(m.sum()), 'truth': int(tb.sum())})
    thresholds = {name: {f'{t:g}m': float(np.mean(np.abs(arr[common] - gt[common]) >= t))
                         for t in (0.5, 1, 2, 3, 5, 10, 20)}
                  for name, arr in [('ZED SDK', sdk), ('DepthNet', dn)]}

    f1 = fig_depth(disp, gt, sdk, dn, target, common)
    f2 = fig_scatter(gt, sdk, dn, common, ms_c, md_c)
    f3 = fig_bins(bins, ['ZED SDK', 'DepthNet'])
    f4 = fig_thresholds(gt, sdk, dn, common, thresholds)
    f5 = fig_headline(ms_c, md_c, ms, md)
    f6 = fig_process()

    def md_table(headers, rows):
        return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                          '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                         ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows])

    main_rows = [
        ['ZED SDK', f"{ms['pixels']:,}", f"{ms['coverage']*100:.2f}%", f"{ms['mae_m']:.3f}",
         f"{ms['rmse_m']:.3f}", f"{ms['bias_m']:+.3f}", f"{ms['median_abs_error_m']:.3f}",
         f"{ms['p90_abs_error_m']:.3f}"],
        ['DepthNet', f"{md['pixels']:,}", f"{md['coverage']*100:.2f}%", f"{md['mae_m']:.3f}",
         f"{md['rmse_m']:.3f}", f"{md['bias_m']:+.3f}", f"{md['median_abs_error_m']:.3f}",
         f"{md['p90_abs_error_m']:.3f}"],
    ]
    common_rows = [
        ['ZED SDK', f"{ms_c['mae_m']:.3f}", f"{ms_c['rmse_m']:.3f}", f"{ms_c['bias_m']:+.3f}",
         f"{ms_c['median_abs_error_m']:.3f}", f"{ms_c['p90_abs_error_m']:.3f}"],
        ['DepthNet', f"{md_c['mae_m']:.3f}", f"{md_c['rmse_m']:.3f}", f"{md_c['bias_m']:+.3f}",
         f"{md_c['median_abs_error_m']:.3f}", f"{md_c['p90_abs_error_m']:.3f}"],
    ]
    bin_rows = []
    for b in bins:
        for name in ('ZED SDK', 'DepthNet'):
            x = b[name]
            bin_rows.append([name, f"{b['lo']}–{b['hi']}", f"{b['common']:,}",
                             f"{x['coverage']*100:.1f}%", f"{x['mae_m']:.3f}",
                             f"{x['rmse_m']:.3f}", f"{x['bias_m']:+.3f}",
                             f"{x['p90_abs_error_m']:.3f}"])
    th_rows = [[name] + [f"{thresholds[name][k]*100:.1f}%" for k in
                         ('0.5m', '1m', '2m', '3m', '5m', '10m', '20m')]
               for name in ('ZED SDK', 'DepthNet')]

    report = f"""# ZED SDK Three-Way Result — PERFORMANCE Mode

**Run (UTC):** 2026-10-28
**Driver:** NVIDIA **621.01** · NVENC H.264/H.265 verified
**SDK:** ZED SDK **5.5.1**, `pyzed` **5.5.1**, official sim runtime **5.5.1**
**Verdict:** **PASS — real ZED SDK depth capture and three-way comparison (PERFORMANCE mode, H.264 stream).**

> This is a genuine ZED SDK result: the SDK opened the virtual ZED XOne GS,
> received the rectified pair, and computed depth over **{ms['pixels']:,} valid
> pixels ({ms['coverage']*100:.2f}% of truth)**. It is **PERFORMANCE** mode (not
> NEURAL) and the stream is **H.264 at 3000 kbps**, because of the workarounds
> below. It is not the kit's original NEURAL/H.265/16000 configuration.

---

## 1. Headline result

{md_table(['Method', 'Valid px', 'Coverage', 'MAE', 'RMSE', 'Bias', 'Median', 'P90'], main_rows)}

**Same common pixels ({summary.get('common_pixels', 0):,}):**

{md_table(['Method', 'MAE', 'RMSE', 'Bias', 'Median', 'P90'], common_rows)}

On the **same SDK-received image** (after correcting the saved file's R/B
swap, PSNR 28.80 dB vs the prepared left), DepthNet has the lower MAE/RMSE:
**{md_c['mae_m']:.3f} m / {md_c['rmse_m']:.3f} m** versus the SDK's
**{ms_c['mae_m']:.3f} m / {ms_c['rmse_m']:.3f} m**. The SDK has the lower
median error (**{ms_c['median_abs_error_m']:.3f} m** vs
**{md_c['median_abs_error_m']:.3f} m**) but a much heavier tail
(P90 {ms_c['p90_abs_error_m']:.3f} m vs {md_c['p90_abs_error_m']:.3f} m;
≥3 m rate {ms_c['bad_3m']*100:.1f}% vs {md_c['bad_3m']*100:.1f}%). The SDK
covers {ms['coverage']*100:.2f}% of truth; DepthNet covers 100%. This is a
mixed result, not a single-method win.

The original `sdk_left.png` was written with a BGRA/RGB channel swap; that
inflated DepthNet's error to 9.34 m. The corrected file is
`outputs/sdk_three_way/sdk_left_corrected.png`; the SDK depth is unchanged.

The earlier **clean-image** DepthNet result is **MAE 2.530 m** on the prepared
left (`STEREO_DEPTH_VALIDATION.md`). The corrected SDK-image DepthNet 2.211 m
and clean 2.530 m are different inputs; do not treat them as one benchmark.
The SDK's 2.800 m sits between them and is not a like-for-like replacement for
the clean-image number.

---

## 2. What was actually run

```powershell
# driver 621.01, NVENC verified
ffmpeg -f lavfi -i color=c=black:s=960x600:d=1 -c:v h264_nvenc -f null NUL

# SDK consumer + runtime
$env:ZED_SDK_ROOT_DIR = 'L:\\save\\userdata\\Downloads\\ZED_SDK_extract\\app'
.venv\\Scripts\\python.exe analysis\\sdk_preflight.py --sdk-root 'L:\\save\\userdata\\Downloads\\ZED_SDK_extract\\app'
.venv\\Scripts\\python.exe scripts\\sim_streamer.py --dll zed_runtime\\sl_zed64.dll --probe

# bridge: PERFORMANCE mode, patched producer (see §3)
.venv\\Scripts\\python.exe analysis\\run_sdk_with_root.py `
  --dll zed_runtime\\sl_zed64.dll --output outputs\\sdk_three_way --depth-mode PERFORMANCE

# comparison
.venv\\Scripts\\python.exe scripts\\compare.py `
  --image outputs\\sdk_three_way\\sdk_left.png `
  --gt outputs\\sdk_three_way\\ue_z_m.npy `
  --sdk-depth outputs\\sdk_three_way\\sdk_z_m.npy `
  --device 0 --output outputs\\sdk_three_way\\comparison
```

---

## 3. Why it failed before, and the exact fix

The failure was never the driver or the camera calibration. It was the
**experimental simulation producer** on Windows. Stereolabs' own issue tracker
documents the same class of failure:

- **[Issue #63 — Isaac Sim 5.1 / ext 4.3.0, Windows, RTX 3090](https://github.com/stereolabs/zed-isaac-sim/issues/63)**
  reports that the bundled `sl_zed64.dll` needs additional
  `StreamingParameters` fields after `chunk_size` (`adaptative_bitrate`,
  `enable_downscale`, `target_width`, `target_height`, `target_bitrate`), plus
  `chunk_size ≤ 1200`, an H.264 fallback, and an explicit GPU-0 CUDA context:
  *“Without this, `initStreamer` failed with NVENC errors.”*
- **[Issue #61 — 5.3 changelog](https://github.com/stereolabs/zed-isaac-sim/issues/61)**
  says the first frame of a network stereo stream can log a red
  `ENCODING_FAILED`, and 5.3 changed it so it no longer logs. In other words,
  **the first `ENCODING_FAILED` is not fatal**; the kit's producer was treating
  it as fatal and exiting before a good frame.
- The official repo's **[`build.bat`](https://github.com/stereolabs/zed-isaac-sim/blob/f9160973f321ec743f626188244cd9702d85bec6/build.bat)**
  pins `SL_ZED_VERSION=5.2.0` and downloads the runtime from
  `.../utils/zed_isaac_sim/sl_zed/5.2.0/...`. The kit's own `ZED_README.md`
  warns that this 5.2.0-labelled dependency and the 5.5 headers may not match.
- The receiver log confirms a **protocol metadata mismatch**:
  `[Streaming] Incorrect metadata size. Expecting: 2904 got 1216. Skipping.`
  The runtime sends 1,216-byte camera metadata; the 5.5.1 consumer expects
  2,904 bytes. The consumer skips the camera settings but still opens with the
  K/baseline from the config; depth is computed on the actual frames.

The successful producer/consumer configuration was:

| Setting | Value | Why |
|---|---|---|
| `chunk_size` | **1200** | Issue #63 Windows/NVENC workaround |
| `target_width/height/bitrate` | **960 / 600 / 3000** | Extended struct fields, issue #63 |
| Codec | **H.264, 3000 kbps** | H.265 fallback in issue #63; metadata/protocol more robust |
| Transport | **BOTH (2)** | Official default; IPC + RTP |
| Timestamp | **stream-relative** | Avoid epoch PTS rejection |
| First `ENCODING_FAILED` | **ignored for first 10 frames** | Issue #61: known first-frame bug |
| Depth mode | **PERFORMANCE** | NEURAL needs TensorRT DLLs on the consumer search path |

The kit's canonical `scripts/sim_streamer.py` and `scripts/run_sdk.py` were
**restored** after the run; the experimental versions are preserved as
`analysis/sim_streamer_success_experimental.py` and
`analysis/run_sdk_success_experimental.py`. `scripts/verify_manifest.py`
reports **461/461**.

---

## 4. Figures

![SDK depth panels](outputs/sdk_three_way/figures/fig01_depth_panels.png)

*Figure 1 — SDK-received left (BGRA corrected for display), UE truth, SDK depth,
DepthNet on the same image, and both error maps. The SDK depth is dense on the
ground and cones; the yellow band at the horizon is the 20 m display cap.*

![Scatter](outputs/sdk_three_way/figures/fig02_scatter.png)

*Figure 2 — Prediction vs truth on the same {summary.get('common_pixels', 0):,}
common pixels.*

![Bins](outputs/sdk_three_way/figures/fig03_bins.png)

*Figure 3 — Accuracy as a function of true depth.*

![Thresholds](outputs/sdk_three_way/figures/fig04_thresholds.png)

*Figure 4 — Cumulative error and bad-pixel rate.*

![Headline](outputs/sdk_three_way/figures/fig05_headline.png)

*Figure 5 — Headline metrics on the same pixels.*

![Process](outputs/sdk_three_way/figures/fig06_process.png)

*Figure 6 — The exact chain that made the SDK capture work.*

The kit's own comparison view is
`outputs/sdk_three_way/comparison_corrected/comparison.png`; it uses the
R/B-corrected SDK left image, so DepthNet is evaluated fairly. The original
`comparison/comparison.png` used the channel-swapped file and is superseded.

---

## 5. Per-depth-bin metrics

{md_table(['Method', 'Truth bin (m)', 'Common px', 'Coverage', 'MAE (m)', 'RMSE (m)', 'Bias (m)', 'P90 abs (m)'], bin_rows)}

## 6. Thresholds (share of common pixels at or above)

{md_table(['Method', '0.5 m', '1 m', '2 m', '3 m', '5 m', '10 m', '20 m'], th_rows)}

---

## 7. Caveats — read before using the number

1. **PERFORMANCE mode, not NEURAL.** The kit's intended mode is NEURAL. The
   extracted SDK's TensorRT DLLs are not on the consumer's search path, so
   NEURAL stops with `NEURAL TRT NOT FOUND`; PERFORMANCE is a real but
   deprecated ZED SDK mode. Do not label this NEURAL.
2. **H.264 3000 kbps, not H.265 16000.** The H.264/3000 fallback is the issue
   #63 workaround. The stream is lossy and the SDK-received left image differs
   from the clean prepared left (PSNR ≈ 20.5 dB). That is why DepthNet on this
   image is much worse than its clean-image 2.530 m.
3. **Metadata mismatch.** The producer runtime sends 1,216-byte camera metadata;
   the 5.5.1 consumer expects 2,904 and skips it. Exposure/gain settings are not
   applied; K and baseline pass the kit's checks. This is the 5.2.0-runtime /
   5.5-consumer version gap the kit warns about.
4. **Static frame.** The same pair is repeated; the SDK logs duplicate-frame
   warnings and requests an IDR. One SDK capture is used.
5. **One frame, synthetic scene.** No temporal stability, noise, rolling
   shutter, or real ZED stereo errors.
6. **`sdk_capture.json` codec string is stale.** The kit hardcodes
   `"H265, 16000 kbps (lossy)"`; the actual producer for this capture used
   H.264 / 3000 kbps / chunk 1200 / transport BOTH. The rest of the capture
   metadata (SDK 5.5.1, PERFORMANCE, K, baseline, timestamp) is accurate.

---

## 8. Bottom line

A genuine ZED SDK 5.5.1 depth capture is now available for the UE pair:
**{ms['pixels']:,} valid pixels, {ms['coverage']*100:.2f}% coverage, MAE
{ms['mae_m']:.3f} m, RMSE {ms['rmse_m']:.3f} m, bias {ms['bias_m']:+.3f} m**.
On the same corrected image, DepthNet has the lower MAE/RMSE
({md_c['mae_m']:.3f} / {md_c['rmse_m']:.3f} m) and full coverage, while the SDK
has the lower median ({ms_c['median_abs_error_m']:.3f} m vs
{md_c['median_abs_error_m']:.3f} m), a heavier tail (P90
{ms_c['p90_abs_error_m']:.3f} m vs {md_c['p90_abs_error_m']:.3f} m) and
{ms['coverage']*100:.2f}% coverage. The clean-image DepthNet result is 2.530 m,
the clean-image SGBM result is 0.791 m at 73.0% coverage; those are different
inputs and are not ranked against the SDK.

This is a real SDK result with documented mode/bitrate caveats, **not** a
NEURAL-mode result and **not** a substitute. The remaining upgrade path is to
copy the SDK's TensorRT DLLs to the consumer search path and retest NEURAL, and
to obtain a protocol-matched 5.5.1 runtime (the published 5.2.0 runtime sends
1,216-byte metadata; the consumer expects 2,904).

### Proof / references

- [Stereolabs zed-isaac-sim repository](https://github.com/stereolabs/zed-isaac-sim)
- [Issue #63 — struct fields, chunk ≤1200, H.264 fallback](https://github.com/stereolabs/zed-isaac-sim/issues/63)
- [Issue #61 — first-frame ENCODING_FAILED fixed in 5.3](https://github.com/stereolabs/zed-isaac-sim/issues/61)
- [Pinned `build.bat` — `SL_ZED_VERSION=5.2.0` runtime URL](https://github.com/stereolabs/zed-isaac-sim/blob/f9160973f321ec743f626188244cd9702d85bec6/build.bat)
- Local proof: `outputs/logs/sdk_h264_3000.log` (metadata mismatch and IDR),
  `outputs/sdk_three_way/sdk_capture.json`, `comparison/summary.json`.
"""
    (ROOT / 'ZED_SDK_SIMULATION_RESULT.md').write_text(report, encoding='utf-8')
    # machine-readable
    (SDK / 'results.json').write_text(json.dumps({
        'sdk': ms, 'depthnet_same_image': md, 'sdk_common': ms_c,
        'depthnet_common': md_c, 'common_pixels': int(common.sum()),
        'bins': bins, 'thresholds': thresholds,
        'capture': cap,
    }, indent=2, allow_nan=False), encoding='utf-8')
    print('Wrote', ROOT / 'ZED_SDK_SIMULATION_RESULT.md')
    print('SDK', ms)
    print('DepthNet same image', md)


if __name__ == '__main__':
    main()

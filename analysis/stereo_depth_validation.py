"""Independent stereo-depth validation for the FSAI 3070 Windows kit.

Runs on the kit data only:
  * re-rectifies the UE stereo pair,
  * runs the bundled DepthNet on the rectified left image (GPU if available),
  * runs an OpenCV SGBM stereo baseline on the rectified pair,
  * compares both with the UE optical-Z truth on identical pixels,
  * writes machine-readable results, figures and the Markdown report.

No result is scaled, shifted or fitted to the ground truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'outputs' / 'yolo_config'))

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image

from compare import evaluate, infer, load_depth
from prepare_stereo import K_TARGET, SIZE, prepare

OUT = ROOT / 'outputs' / 'stereo_validation'
FIG = OUT / 'figures'
TABLES = OUT / 'tables'
PREPARED = OUT / 'prepared'

# ---- design tokens (dataviz reference palette, light surface) -------------
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
BLUE = '#2a78d6'
ORANGE = '#eb6834'
GRAY = '#898781'
BAD_FILL = '#d9d8d2'

DN_COLOR = BLUE
SG_COLOR = ORANGE

DEPTH_CMAP = LinearSegmentedColormap.from_list(
    'depth_blue', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf',
                   '#184f95', '#0d366b'])
ERROR_CMAP = LinearSegmentedColormap.from_list(
    'error_orange', ['#fde7dc', '#f7b799', '#ee8558', '#d95926', '#a63c14', '#6f260b'])
DIVERGE_CMAP = LinearSegmentedColormap.from_list(
    'bias_blue_red', ['#184f95', '#6da7ec', '#f0efec', '#f0a3a2', '#d03b3b'])

plt.rcParams.update({
    'font.family': ['Segoe UI', 'DejaVu Sans'],
    'font.size': 10,
    'axes.facecolor': SURFACE,
    'figure.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS,
    'axes.linewidth': 0.8,
    'axes.labelcolor': INK2,
    'axes.titlecolor': INK,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.grid': True,
    'grid.color': GRID,
    'grid.linewidth': 0.6,
    'grid.linestyle': '-',
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'text.color': INK,
    'legend.frameon': False,
    'legend.fontsize': 9,
    'lines.linewidth': 2.0,
    'lines.markersize': 4,
})


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def metric_block(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {'pixels': 0, 'mae_m': None, 'rmse_m': None, 'bias_m': None,
                'median_abs_error_m': None, 'p90_abs_error_m': None,
                'p95_abs_error_m': None, 'max_abs_error_m': None,
                'abs_rel': None, 'pearson_r': None, 'r2': None,
                'delta_1_25': None, 'delta_1_25_2': None, 'delta_1_25_3': None}
    g = gt[mask].astype(np.float64)
    p = pred[mask].astype(np.float64)
    d = p - g
    ad = np.abs(d)
    ratio = np.maximum(p / g, g / p)
    if g.size > 1 and g.std() > 0 and p.std() > 0:
        pearson = float(np.corrcoef(g, p)[0, 1])
        r2 = float(1.0 - np.sum(d ** 2) / np.sum((g - g.mean()) ** 2))
    else:
        pearson, r2 = None, None
    return {
        'pixels': n,
        'mae_m': float(ad.mean()),
        'rmse_m': float(np.sqrt(np.mean(d ** 2))),
        'bias_m': float(d.mean()),
        'median_abs_error_m': float(np.median(ad)),
        'p90_abs_error_m': float(np.percentile(ad, 90)),
        'p95_abs_error_m': float(np.percentile(ad, 95)),
        'max_abs_error_m': float(ad.max()),
        'abs_rel': float(np.mean(ad / g)),
        'pearson_r': pearson,
        'r2': r2,
        'delta_1_25': float(np.mean(ratio < 1.25)),
        'delta_1_25_2': float(np.mean(ratio < 1.25 ** 2)),
        'delta_1_25_3': float(np.mean(ratio < 1.25 ** 3)),
    }


def evaluate_protocol(gt: np.ndarray, predictions: dict[str, np.ndarray],
                      minimum: float, maximum: float) -> dict:
    target = np.isfinite(gt) & (gt > minimum) & (gt <= maximum)
    masks = {}
    common = target.copy()
    for name, pred in predictions.items():
        if pred.shape != gt.shape:
            raise ValueError(f'{name}: {pred.shape} != truth {gt.shape}')
        masks[name] = target & np.isfinite(pred) & (pred > 0)
        common &= masks[name]
    result = {
        'gt_min_exclusive_m': minimum,
        'gt_max_inclusive_m': maximum,
        'target_pixels': int(target.sum()),
        'common_pixels': int(common.sum()),
        'pairwise': {},
        'common_pixel_metrics': {},
    }
    for name, pred in predictions.items():
        m = metric_block(gt, pred, masks[name])
        m['coverage_of_gt'] = float(masks[name].sum() / target.sum())
        result['pairwise'][name] = m
        result['common_pixel_metrics'][name] = metric_block(gt, pred, common)
    result['masks'] = {'target': target, 'common': common, **masks}
    return result


def bad_rate(gt, pred, mask, threshold):
    if not mask.any():
        return None
    return float(np.mean(np.abs(pred[mask] - gt[mask]) >= threshold))


def bin_metrics(gt, predictions, masks, common, edges):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = common & (gt >= lo) & (gt < hi)
        target_b = masks['target'] & (gt >= lo) & (gt < hi)
        n = int(b.sum())
        for name, pred in predictions.items():
            row = {'method': name, 'bin_lo_m': float(lo), 'bin_hi_m': float(hi),
                   'gt_pixels_in_bin': int(target_b.sum()),
                   'common_pixels': n,
                   'coverage_of_gt': float(masks[name][target_b].sum() / target_b.sum())
                   if target_b.any() else None}
            if n:
                g, p = gt[b].astype(np.float64), pred[b].astype(np.float64)
                d = p - g
                row.update({
                    'mae_m': float(np.abs(d).mean()),
                    'rmse_m': float(np.sqrt(np.mean(d ** 2))),
                    'bias_m': float(d.mean()),
                    'p90_abs_error_m': float(np.percentile(np.abs(d), 90)),
                })
            else:
                row.update({'mae_m': None, 'rmse_m': None, 'bias_m': None,
                            'p90_abs_error_m': None})
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# stereo
# ---------------------------------------------------------------------------
def make_sgbm(block_size: int, num_disparities: int, uniqueness: int):
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        preFilterCap=63,
        uniquenessRatio=uniqueness,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def run_sgbm(gray_l, gray_r, block_size=5, num_disparities=160, uniqueness=10):
    # disp12MaxDiff=1 makes SGBM perform its own left-right consistency check,
    # so no second right-view matcher is needed (and the right-view map from
    # this OpenCV build is not a reliable independent check).
    matcher = make_sgbm(block_size, num_disparities, uniqueness)
    disp = matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
    valid = np.isfinite(disp) & (disp > 0)
    return disp, valid


def disparity_to_depth(disp, f, baseline):
    z = np.full(disp.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disp) & (disp > 0)
    z[valid] = f * baseline / disp[valid]
    return z


def segment_cones(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    mask = ((sat > 0.45) & (val > 0.25)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    regions = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 40:
            continue
        regions.append({'id': len(regions) + 1, 'component': i, 'area_px': area,
                        'x': int(stats[i, cv2.CC_STAT_LEFT]),
                        'y': int(stats[i, cv2.CC_STAT_TOP]),
                        'w': int(stats[i, cv2.CC_STAT_WIDTH]),
                        'h': int(stats[i, cv2.CC_STAT_HEIGHT]),
                        'cx': float(centroids[i][0]), 'cy': float(centroids[i][1])})
    return labels, regions


def region_metrics(labels, regions, gt, predictions, common):
    rows = []
    for r in regions:
        region = labels == r['component']
        gvals = gt[region & np.isfinite(gt) & (gt > 0) & (gt <= 20)]
        r['gt_median_m'] = float(np.median(gvals)) if gvals.size else None
        r['gt_pixels'] = int(gvals.size)
        for name, pred in predictions.items():
            m = region & common
            row = {'region_id': r['id'], 'method': name, 'area_px': r['area_px'],
                   'gt_median_m': r['gt_median_m'], 'common_pixels': int(m.sum())}
            block = metric_block(gt, pred, m)
            row.update({k: block[k] for k in ('mae_m', 'rmse_m', 'bias_m',
                                              'p90_abs_error_m')})
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------
def save_fig(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=170, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    return path


def depth_panel(ax, arr, valid, title, vmax=20.0, cmap=DEPTH_CMAP, cbar=True,
                cbar_label='Depth (m)'):
    cm = cmap.copy()
    cm.set_bad(BAD_FILL)
    im = ax.imshow(np.ma.array(arr, mask=~valid), cmap=cm, vmin=0, vmax=vmax,
                   interpolation='nearest')
    ax.set_title(title, loc='left')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
        cb.set_label(cbar_label, color=INK2, fontsize=8)
        cb.ax.tick_params(labelsize=7, colors=MUTED)
        cb.outline.set_edgecolor(AXIS)
    return im


def fig_inputs(left, right, meta):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    for ax, img, title in [(axes[0], left, 'Rectified left'),
                           (axes[1], right, 'Rectified right')]:
        ax.imshow(img)
        ax.set_title(title, loc='left')
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        for y in (150, 300, 450):
            ax.axhline(y, color='#ffffff', lw=0.8, alpha=0.85)
            ax.axhline(y, color=INK, lw=0.35, alpha=0.35)
        ax.text(8, 158, 'same row', color='#ffffff', fontsize=7, alpha=0.95,
                bbox=dict(facecolor=INK, alpha=0.45, pad=1.5, edgecolor='none'))
    fig.suptitle('Rectified stereo pair — 960 × 600, f = 636.25 px, '
                 f'B = {meta["baseline_m"]*1000:.3f} mm', color=INK, fontsize=12,
                 fontweight='bold')
    return save_fig(fig, 'fig01_input_pair.png')


def fig_depth_comparison(left, gt, dn, sg, masks, common, metrics):
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.4), layout='constrained')
    axes[0, 0].imshow(left)
    axes[0, 0].set_title('Rectified left input', loc='left')
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([]); axes[0, 0].grid(False)
    for s in axes[0, 0].spines.values():
        s.set_visible(False)
    im_gt = depth_panel(axes[0, 1], gt, masks['target'], 'UE optical-Z truth (m)',
                        cbar=False)
    depth_panel(axes[0, 2], dn, masks['DepthNet'],
                f'DepthNet (m) — MAE {metrics["DepthNet"]["mae_m"]:.2f}', cbar=False)
    depth_panel(axes[1, 0], sg, masks['SGBM'],
                f'SGBM stereo (m) — MAE {metrics["SGBM"]["mae_m"]:.2f}', cbar=False)
    depth_panel(axes[1, 1], np.abs(dn - gt), common,
                'DepthNet |error| (m)', vmax=5, cmap=ERROR_CMAP, cbar=False)
    err_sg = depth_panel(axes[1, 2], np.abs(sg - gt), common,
                         'SGBM |error| (m)', vmax=5, cmap=ERROR_CMAP, cbar=False)
    cb1 = fig.colorbar(im_gt, ax=[axes[0, 1], axes[0, 2], axes[1, 0]],
                       fraction=0.025, pad=0.012)
    cb1.set_label('Depth (m)', color=INK2, fontsize=8)
    cb1.ax.tick_params(labelsize=7, colors=MUTED)
    cb1.outline.set_edgecolor(AXIS)
    cb2 = fig.colorbar(err_sg, ax=[axes[1, 1], axes[1, 2]],
                       fraction=0.04, pad=0.012)
    cb2.set_label('Absolute error (m)', color=INK2, fontsize=8)
    cb2.ax.tick_params(labelsize=7, colors=MUTED)
    cb2.outline.set_edgecolor(AXIS)
    fig.suptitle('Stereo depth against UE optical-Z truth (0 < Z ≤ 20 m, common pixels for errors)',
                 color=INK, fontsize=12, fontweight='bold')
    return save_fig(fig, 'fig02_depth_comparison.png')


def fig_scatter(gt, predictions, common, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    for ax, (name, pred), color, cmap in [
            (axes[0], list(predictions.items())[0], DN_COLOR, DEPTH_CMAP),
            (axes[1], list(predictions.items())[1], SG_COLOR, ERROR_CMAP)]:
        m = common
        hb = ax.hexbin(gt[m], pred[m], gridsize=55, mincnt=2,
                       cmap=cmap, linewidths=0, extent=(0, 20, 0, 20))
        counts = np.asarray(hb.get_array(), dtype=float)
        vmax = max(1.0, float(np.nanmax(counts)) if counts.size else 1.0)
        hb.set_norm(LogNorm(vmin=1.0, vmax=vmax))
        ax.plot([0, 20], [0, 20], color=INK2, lw=1.0, ls='-', alpha=0.8)
        ax.set_xlim(0, 20); ax.set_ylim(0, 20)
        ax.set_xlabel('UE optical-Z truth (m)')
        ax.set_ylabel('Predicted depth (m)')
        mm = metrics[name]
        ax.set_title(f'{name} — MAE {mm["mae_m"]:.2f} m · '
                     f'median |err| {mm["median_abs_error_m"]:.2f} m',
                     loc='left')
        ax.text(0.98, 0.04, f'{mm["pixels"]:,} common px', transform=ax.transAxes,
                ha='right', va='bottom', color=INK2, fontsize=8)
        cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label('Pixels per bin (log)', color=INK2, fontsize=8)
        cb.ax.tick_params(labelsize=7, colors=MUTED)
        cb.outline.set_edgecolor(AXIS)
    fig.suptitle('Prediction vs truth on identical pixels; gray line is y = x',
                 color=INK, fontsize=12, fontweight='bold')
    return save_fig(fig, 'fig03_scatter.png')


def fig_error_vs_depth(bin_rows, predictions):
    methods = list(predictions)
    bins = sorted({(r['bin_lo_m'], r['bin_hi_m']) for r in bin_rows})
    centers = np.array([(lo + hi) / 2 for lo, hi in bins])
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), layout='constrained', sharex=True)
    specs = [('mae_m', 'MAE (m)', False), ('rmse_m', 'RMSE (m)', False),
             ('p90_abs_error_m', 'P90 absolute error (m)', False),
             ('coverage_of_gt', 'Coverage of truth pixels (%)', True),
             ('bias_m', 'Signed bias, pred − truth (m)', False),
             ('common_pixels', 'Common pixels in bin', False)]
    colors = {methods[0]: DN_COLOR, methods[1]: SG_COLOR} if len(methods) > 1 else {
        methods[0]: DN_COLOR}
    for ax, (key, label, pct) in zip(axes.flat, specs):
        for name in methods:
            vals = []
            for lo, hi in bins:
                r = next(x for x in bin_rows
                         if x['method'] == name and x['bin_lo_m'] == lo
                         and x['bin_hi_m'] == hi)
                v = r.get(key)
                if v is None:
                    vals.append(np.nan)
                elif pct:
                    vals.append(v * 100)
                else:
                    vals.append(v)
            ax.plot(centers, vals, marker='o', color=colors[name], label=name,
                    markeredgecolor=SURFACE, markeredgewidth=1.0)
        ax.set_title(label, loc='left')
        ax.set_xlabel('UE truth bin centre (m)')
        ax.set_xticks(centers)
        ax.tick_params(axis='x', labelrotation=45, labelsize=8)
        ax.grid(True, axis='y')
        if key in ('common_pixels',):
            ax.set_yscale('log')
        if key == 'bias_m':
            ax.axhline(0, color=AXIS, lw=0.8)
        ax.legend(loc='best')
    axes[0, 0].set_ylabel('metres')
    fig.suptitle('Accuracy as a function of true depth (common pixels within each bin)',
                 color=INK, fontsize=12, fontweight='bold')
    return save_fig(fig, 'fig04_error_vs_depth.png')


def fig_error_distribution(gt, predictions, common):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    bins = np.arange(0, 5.01, 0.1)
    colors = {list(predictions)[0]: DN_COLOR, list(predictions)[1]: SG_COLOR}
    tail_text = []
    for name, pred in predictions.items():
        err = np.abs(pred[common] - gt[common])
        axes[0].hist(np.clip(err, 0, 5), bins=bins, density=True, histtype='step',
                     lw=1.8, color=colors[name], label=name)
        xs = np.logspace(-3, 2, 700)
        cdf = [np.mean(err <= x) * 100 for x in xs]
        axes[1].plot(xs, cdf, color=colors[name], label=name)
        pct = np.mean(err <= 2) * 100
        axes[1].annotate(f'{pct:.1f}% ≤ 2 m', (2, pct), xytext=(6, 6),
                         textcoords='offset points', fontsize=8,
                         color=colors[name], fontweight='bold')
        tail_text.append(f'{name}: {np.mean(err > 5) * 100:.1f}% > 5 m')
    axes[0].set_xlabel('Absolute error (m), last bin includes ≥ 5 m')
    axes[0].set_ylabel('Density (log scale)')
    axes[0].set_yscale('log')
    axes[0].set_title('Absolute-error distribution', loc='left')
    axes[0].legend(loc='upper right')
    axes[1].set_xlabel('Absolute-error threshold (m, log scale)')
    axes[1].set_ylabel('Share of common pixels ≤ threshold (%)')
    axes[1].set_title('Cumulative error curve — including the tail', loc='left')
    axes[1].set_xscale('log')
    axes[1].set_xlim(1e-3, 100); axes[1].set_ylim(0, 100)
    axes[1].legend(loc='lower right')
    axes[1].text(0.02, 0.55, '\n'.join(tail_text), transform=axes[1].transAxes,
                 ha='left', va='bottom', fontsize=8, color=INK2,
                 bbox=dict(facecolor=SURFACE, edgecolor=AXIS, lw=0.5, pad=3))
    fig.suptitle('Where the error lives — identical common pixels for both methods',
                 color=INK, fontsize=12, fontweight='bold')
    return save_fig(fig, 'fig05_error_distribution.png')


def fig_metrics(protocol, bad_rates):
    methods = list(protocol['pairwise'])
    colors = {methods[0]: DN_COLOR, methods[1]: SG_COLOR}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), layout='constrained')
    groups = [('MAE', 'mae_m', 'm'), ('RMSE', 'rmse_m', 'm')]
    x = np.arange(len(groups))
    width = 0.34
    for i, name in enumerate(methods):
        vals = [protocol['common_pixel_metrics'][name][k] for _, k, _ in groups]
        bars = axes[0].bar(x + (i - 0.5) * width, vals, width * 0.92,
                           color=colors[name], label=name)
        for b, v in zip(bars, vals):
            axes[0].text(b.get_x() + b.get_width() / 2, v, f'{v:.2f}',
                         ha='center', va='bottom', fontsize=8, color=INK2)
    axes[0].set_xticks(x, [g[0] for g in groups])
    axes[0].set_ylabel('metres')
    axes[0].set_title('Error on common pixels', loc='left')
    axes[0].legend(loc='upper left')

    for ax, key, label in [
            (axes[1], 'coverage_of_gt', 'Coverage of truth (%)'),
            (axes[2], 'bad_3m', 'Pixels with |error| ≥ 3 m (%)')]:
        vals = []
        for name in methods:
            v = protocol['pairwise'][name]['coverage_of_gt'] * 100 if key == 'coverage_of_gt' \
                else bad_rates[name] * 100
            vals.append(v)
        bars = ax.bar(np.arange(len(methods)), vals, 0.55,
                      color=[colors[m] for m in methods])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.1f}%',
                    ha='center', va='bottom', fontsize=8, color=INK2)
        ax.set_xticks(np.arange(len(methods)), methods)
        ax.set_ylim(0, 105)
        ax.set_title(label, loc='left')
    fig.suptitle('Headline metrics — pairwise coverage, common-pixel error',
                 color=INK, fontsize=12, fontweight='bold')
    return save_fig(fig, 'fig06_headline_metrics.png')


def fig_regions(left, labels, regions, gt, predictions, common):
    display = left.copy()
    for r in regions:
        region = (labels == r['component']).astype(np.uint8)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, contours, -1, (11, 11, 11), 1, cv2.LINE_AA)
        x, y = int(r['x']), int(r['y'])
        cv2.putText(display, str(r['id']), (x, max(10, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (11, 11, 11), 1, cv2.LINE_AA)
    selected = []
    for r in regions:
        if r.get('gt_median_m') is None:
            continue
        region = labels == r['component']
        if int((region & common).sum()) >= 10:
            selected.append(r)
    selected.sort(key=lambda r: r['gt_median_m'])
    selected = selected[:16]
    rows = []
    for r in selected:
        region = labels == r['component']
        for name, pred in predictions.items():
            m = region & common
            if m.any():
                rows.append((r['id'], r['gt_median_m'], name,
                             float(np.mean(np.abs(pred[m] - gt[m])))))
    names = list(predictions)
    colors = {names[0]: DN_COLOR, names[1]: SG_COLOR}
    ids = [r['id'] for r in selected]
    y = np.arange(len(ids))
    fig = plt.figure(figsize=(12.5, max(4.5, 0.34 * len(ids) + 1.8)),
                     layout='constrained')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(display)
    ax0.set_title('Detected cone regions (saturation segmentation)', loc='left')
    ax0.set_xticks([]); ax0.set_yticks([]); ax0.grid(False)
    for s in ax0.spines.values():
        s.set_visible(False)
    ax1 = fig.add_subplot(gs[0, 1])
    height = 0.38
    for i, name in enumerate(names):
        vals = []
        for rid in ids:
            v = next((x[3] for x in rows if x[0] == rid and x[2] == name), np.nan)
            vals.append(v)
        bars = ax1.barh(y + (i - 0.5) * height, vals, height * 0.9,
                        color=colors[name], label=name)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax1.text(v, b.get_y() + b.get_height() / 2, f' {v:.2f}',
                         va='center', fontsize=7, color=INK2)
    ax1.set_yticks(y, [f'R{rid} · {next(r["gt_median_m"] for r in selected if r["id"] == rid):.1f} m'
                       for rid in ids])
    ax1.invert_yaxis()
    ax1.set_xlabel('MAE on region (m)')
    ax1.set_title('Per-cone MAE (regions with ≥ 10 common px)', loc='left')
    ax1.legend(loc='lower right')
    fig.suptitle('Region-level check — cones are the only strongly textured objects',
                 color=INK, fontsize=12, fontweight='bold')
    return save_fig(fig, 'fig07_cone_regions.png')


def fig_disparity(gt, sg_disp, common, f, baseline, stats):
    expected = np.full(gt.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(gt) & (gt > 0) & (gt <= 20)
    expected[valid] = f * baseline / gt[valid]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), layout='constrained')
    depth_panel(axes[0], expected, valid, 'Truth-implied disparity (px)',
                vmax=80, cbar_label='Disparity (px)')
    depth_panel(axes[1], sg_disp, np.isfinite(sg_disp) & (sg_disp > 0),
                'SGBM disparity (px)', vmax=80, cbar_label='Disparity (px)')
    diff = sg_disp - expected
    cm = DIVERGE_CMAP.copy(); cm.set_bad(BAD_FILL)
    im = axes[2].imshow(np.ma.array(diff, mask=~common), cmap=cm, vmin=-10, vmax=10,
                        interpolation='nearest')
    axes[2].set_title('SGBM − truth-implied disparity (px)', loc='left')
    axes[2].set_xticks([]); axes[2].set_yticks([]); axes[2].grid(False)
    for s in axes[2].spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=axes[2], fraction=0.04, pad=0.015)
    cb.set_label('Disparity error (px)', color=INK2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    cb.outline.set_edgecolor(AXIS)
    fig.suptitle(f'Geometric consistency — median |Δd| {stats["median_abs_disparity_px"]:.2f} px, '
                 f'{stats["within_1px_pct"] * 100:.1f}% within 1 px (common pixels)',
                 color=INK, fontsize=12, fontweight='bold')
    return save_fig(fig, 'fig08_disparity_consistency.png')


def fig_epipolar(left, right, dy, dx, stats):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    if len(dy) == 0:
        axes[0].text(0.5, 0.5, 'No SIFT ratio-test matches', ha='center',
                     va='center', transform=axes[0].transAxes, color=INK2)
        axes[0].set_title('Epipolar row alignment — unavailable', loc='left')
        axes[1].set_visible(False)
        return save_fig(fig, 'fig09_epipolar_check.png')
    axes[0].hist(dy, bins=np.arange(-2, 2.01, 0.1), color=BLUE, alpha=0.9)
    axes[0].axvline(0, color=INK2, lw=1.0)
    axes[0].axvline(np.median(dy), color=ORANGE, lw=1.4, ls='-',
                    label=f'median {np.median(dy):+.2f} px')
    axes[0].set_xlabel('Vertical offset y_right − y_left (px)')
    axes[0].set_ylabel('Matched features')
    axes[0].set_title(f'Epipolar row alignment — {len(dy)} SIFT matches', loc='left')
    axes[0].legend(loc='upper right')
    axes[1].scatter(dx, dy, s=10, color=BLUE, alpha=0.55, edgecolors='none')
    axes[1].axhline(0, color=INK2, lw=0.8)
    axes[1].set_xlim(-170, 30)
    n_out = int(np.count_nonzero((dx < -170) | (dx > 30)))
    if n_out:
        axes[1].text(0.98, 0.04, f'{n_out} matches outside x-range',
                     transform=axes[1].transAxes, ha='right', va='bottom',
                     fontsize=7.5, color=INK2)
    axes[1].set_xlabel('Horizontal offset x_right − x_left (px)')
    axes[1].set_ylabel('Vertical offset (px)')
    axes[1].set_title(f'Median |dy| {stats["median_abs_dy_px"]:.2f} px · '
                      f'P95 {stats["p95_abs_dy_px"]:.2f} px', loc='left')
    fig.suptitle('Rectification check — a parallel pair puts every match on the same row',
                 color=INK, fontsize=12, fontweight='bold')
    return save_fig(fig, 'fig09_epipolar_check.png')


def fig_pipeline(meta, details):
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis('off'); ax.grid(False)
    boxes = [
        (0.2, 3.3, 2.3, 1.1, 'Kit data\nstereo_demo L/R RGB\n+ UE optical Z', GRAY),
        (1.0, 1.4, 2.3, 1.0, 'Integrity\nmanifest · unit tests\nreplay', GRAY),
        (3.3, 3.3, 2.5, 1.1, 'Rectify (OpenCV)\n960×600 · f 636.25 px\nB 0.1197 m', INK2),
        (6.5, 4.0, 2.6, 1.0, 'DepthNet\nmono on rectified left\nGPU/CUDA', DN_COLOR),
        (6.5, 2.5, 2.6, 1.0, 'SGBM stereo\nrectified L/R pair\ninternal L/R check', SG_COLOR),
        (9.9, 3.25, 2.7, 1.0, 'Evaluate vs UE Z\n0 < Z ≤ 20 m\ncommon-pixel mask', INK2),
        (9.9, 1.3, 2.7, 1.0, 'Report\nmetrics · tables\nfigures', INK),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06,rounding_size=0.12',
                                    facecolor='#ffffff', edgecolor=color, linewidth=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=9,
                color=INK, linespacing=1.35)
    arrows = [
        ((2.5, 3.85), (3.3, 3.85), INK2),
        ((5.8, 3.85), (6.5, 4.5), DN_COLOR),
        ((5.8, 3.85), (6.5, 3.0), SG_COLOR),
        ((9.1, 4.5), (9.9, 3.85), INK2),
        ((9.1, 3.0), (9.9, 3.65), INK2),
        ((11.25, 3.25), (11.25, 2.3), INK),
        ((3.0, 2.4), (3.8, 3.3), GRAY),
    ]
    for (x1, y1), (x2, y2), color in arrows:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                     mutation_scale=12, linewidth=1.2, color=color,
                                     shrinkA=0, shrinkB=0))
    ax.text(0.2, 0.55, f'DepthNet device: {details["device"]} · shared laptop; '
                       'timing varies with GPU contention (see report §5)',
            fontsize=8.5, color=INK2)
    ax.set_title('Validation process — no SDK depth is substituted', loc='left',
                 fontsize=13, fontweight='bold')
    return save_fig(fig, 'fig10_pipeline.png')


# ---------------------------------------------------------------------------
# report helpers
# ---------------------------------------------------------------------------
def fmt(v, digits=3, pct=False, signed=False):
    if v is None:
        return '—'
    if pct:
        return f'{v*100:.2f}%'
    if signed:
        return f'{v:+.{digits}f}'
    return f'{v:.{digits}f}'


def md_table(headers, rows, align=None):
    align = align or ['---'] * len(headers)
    lines = ['| ' + ' | '.join(headers) + ' |',
             '| ' + ' | '.join(align) + ' |']
    lines += ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows]
    return '\n'.join(lines)


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='0')
    parser.add_argument('--num-disparities', type=int, default=160)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    PREPARED.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)
    timings = {}

    # 1. prepare rectified pair
    t0 = time.perf_counter()
    meta = prepare(ROOT / 'data' / 'stereo_demo', PREPARED)
    timings['rectify_s'] = time.perf_counter() - t0
    left_path, right_path = PREPARED / 'left.png', PREPARED / 'right.png'
    gt_path = PREPARED / 'ue_z_m.npy'
    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
    gt = load_depth(gt_path)
    if left.shape[:2] != gt.shape or right.shape[:2] != gt.shape:
        raise RuntimeError('Prepared image/truth shape mismatch')
    gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # 2. DepthNet on the rectified left image
    t0 = time.perf_counter()
    dn, details = infer(left_path, args.device)
    timings['depthnet_wall_s'] = time.perf_counter() - t0
    try:
        import torch
        details['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        details['torch_cuda_build'] = torch.version.cuda
        if torch.cuda.is_available():
            details['vram_total_gib'] = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            details['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 1024 ** 3
            details['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 1024 ** 3
    except Exception as exc:  # pragma: no cover
        details['gpu_error'] = str(exc)
    if dn.shape != gt.shape:
        raise RuntimeError(f'DepthNet shape {dn.shape} != truth {gt.shape}')
    np.save(OUT / 'depthnet_z_m.npy', dn)

    # 3. SGBM baseline and sensitivity
    t0 = time.perf_counter()
    disp_l, valid_disp = run_sgbm(gray_l, gray_r, block_size=5,
                                  num_disparities=args.num_disparities,
                                  uniqueness=10)
    timings['sgbm_ms'] = (time.perf_counter() - t0) * 1000
    meta['sgbm_ms'] = timings['sgbm_ms']
    f = float(K_TARGET[0, 0])
    baseline = float(meta['baseline_m'])
    sg = disparity_to_depth(disp_l, f, baseline)
    sg[~valid_disp] = np.nan
    sg[(sg < 0.1) | (sg > 80)] = np.nan
    np.save(OUT / 'sgbm_disparity_px.npy', disp_l)
    np.save(OUT / 'sgbm_z_m.npy', sg)

    sensitivity = []
    for bs in (3, 5, 7):
        t = time.perf_counter()
        d, l = run_sgbm(gray_l, gray_r, block_size=bs,
                        num_disparities=args.num_disparities, uniqueness=10)
        z = disparity_to_depth(d, f, baseline)
        z[~l] = np.nan
        m = np.isfinite(gt) & (gt > 0) & (gt <= 20) & np.isfinite(z) & (z > 0) & (z <= 80)
        zz = z.copy(); zz[~m] = np.nan
        # pairwise metrics on this configuration's own valid pixels
        mm = metric_block(gt, z, m)
        sens = {'block_size': bs, 'num_disparities': args.num_disparities,
                'uniqueness': 10, 'valid_pixels': int(m.sum()),
                'coverage_of_gt': float(m.sum() / (np.isfinite(gt) & (gt > 0) & (gt <= 20)).sum()),
                'mae_m': mm['mae_m'], 'rmse_m': mm['rmse_m'],
                'bias_m': mm['bias_m'], 'runtime_ms': (time.perf_counter() - t) * 1000}
        sensitivity.append(sens)
    write_csv(OUT / 'sgbm_sensitivity.csv', sensitivity)

    # 4. metrics: main protocol and extended range
    predictions = {'DepthNet': dn, 'SGBM': sg}
    protocol = evaluate_protocol(gt, predictions, 0.0, 20.0)
    extended = evaluate_protocol(gt, predictions, 0.5, 80.0)
    masks = protocol['masks']
    common = masks['common']

    bad_rates = {name: bad_rate(gt, pred, common, 3.0) for name, pred in predictions.items()}
    thresholds = {}
    for name, pred in predictions.items():
        thresholds[name] = {f'{t:g}m': bad_rate(gt, pred, common, t)
                            for t in (0.5, 1, 2, 3, 5, 10, 20)}

    edges = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    bm = bin_metrics(gt, predictions, {**masks, 'target': masks['target']},
                     common, edges)
    write_csv(OUT / 'depth_bins.csv', bm)

    # disparity consistency
    expected_disp = np.full(gt.shape, np.nan, dtype=np.float32)
    valid_gt = np.isfinite(gt) & (gt > 0) & (gt <= 20)
    expected_disp[valid_gt] = f * baseline / gt[valid_gt]
    dvalid = common & valid_disp
    dd = np.abs(disp_l[dvalid] - expected_disp[dvalid])
    disp_stats = {
        'evaluated_pixels': int(dvalid.sum()),
        'median_abs_disparity_px': float(np.median(dd)),
        'p90_abs_disparity_px': float(np.percentile(dd, 90)),
        'within_1px_pct': float(np.mean(dd <= 1.0)),
        'within_2px_pct': float(np.mean(dd <= 2.0)),
    }

    # epipolar check
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.01)
    kp_l, des_l = sift.detectAndCompute(gray_l, None)
    kp_r, des_r = sift.detectAndCompute(gray_r, None)
    dy = np.array([], dtype=np.float32)
    dx = np.array([], dtype=np.float32)
    if des_l is not None and des_r is not None and len(kp_l) and len(kp_r):
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        knn = matcher.knnMatch(des_l, des_r, k=2)
        good = [m for m, n in knn if m.distance < 0.75 * n.distance]
        dy = np.array([kp_r[m.trainIdx].pt[1] - kp_l[m.queryIdx].pt[1]
                       for m in good], dtype=np.float32)
        dx = np.array([kp_r[m.trainIdx].pt[0] - kp_l[m.queryIdx].pt[0]
                       for m in good], dtype=np.float32)
    epi_stats = {
        'sift_keypoints_left': len(kp_l), 'sift_keypoints_right': len(kp_r),
        'ratio_test_matches': int(len(dy)),
        'median_dy_px': float(np.median(dy)) if len(dy) else None,
        'median_abs_dy_px': float(np.median(np.abs(dy))) if len(dy) else None,
        'p95_abs_dy_px': float(np.percentile(np.abs(dy), 95)) if len(dy) else None,
        'share_abs_dy_le_1px': float(np.mean(np.abs(dy) <= 1.0)) if len(dy) else None,
    }

    # regions
    labels, regions = segment_cones(left)
    region_rows = region_metrics(labels, regions, gt, predictions, common)
    write_csv(OUT / 'cone_regions.csv', region_rows)

    # figures
    figures = {}
    figures['inputs'] = fig_inputs(left_rgb, right_rgb, meta)
    figures['depth'] = fig_depth_comparison(left_rgb, gt, dn, sg, masks, common,
                                            protocol['common_pixel_metrics'])
    figures['scatter'] = fig_scatter(gt, predictions, common, protocol['common_pixel_metrics'])
    figures['bins'] = fig_error_vs_depth(bm, predictions)
    figures['distribution'] = fig_error_distribution(gt, predictions, common)
    figures['headline'] = fig_metrics(protocol, bad_rates)
    figures['regions'] = fig_regions(left_rgb, labels, regions, gt, predictions, common)
    figures['disparity'] = fig_disparity(gt, disp_l, common, f, baseline, disp_stats)
    figures['epipolar'] = fig_epipolar(left, right, dy, dx, epi_stats)
    figures['pipeline'] = fig_pipeline(meta, details)

    # CPU reference comparison for the same prepared left image
    cpu_ref_path = ROOT / 'validation' / 'stereo_net_only' / 'depthnet_z_m.npy'
    cpu_compare = None
    if cpu_ref_path.is_file():
        cpu_ref = load_depth(cpu_ref_path)
        if cpu_ref.shape == dn.shape:
            both = np.isfinite(cpu_ref) & np.isfinite(dn)
            diff = np.abs(dn[both] - cpu_ref[both])
            cpu_compare = {
                'reference': str(cpu_ref_path.relative_to(ROOT)),
                'comparable_pixels': int(both.sum()),
                'max_abs_diff_m': float(diff.max()) if diff.size else None,
                'mean_abs_diff_m': float(diff.mean()) if diff.size else None,
                'bitwise_equal': bool(np.array_equal(dn, cpu_ref)),
            }

    # environment snapshot
    env = {
        'python': sys.version,
        'platform': platform.platform(),
        'torch': details.get('torch'),
        'torch_cuda_build': details.get('torch_cuda_build'),
        'ultralytics': details.get('ultralytics'),
        'device': details.get('device'),
        'gpu': details.get('gpu'),
        'vram_total_gib': details.get('vram_total_gib'),
        'peak_allocated_gib': details.get('peak_allocated_gib'),
        'peak_reserved_gib': details.get('peak_reserved_gib'),
        'cpu': platform.processor(),
        'ram_gib': None,
        'opencv': cv2.__version__,
        'numpy': np.__version__,
        'matplotlib': matplotlib.__version__,
        'nvidia_smi': None,
    }
    try:
        import psutil
        env['ram_gib'] = psutil.virtual_memory().total / 1024 ** 3
    except Exception:
        pass
    try:
        env['nvidia_smi'] = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
             '--format=csv,noheader'], text=True, capture_output=True,
            timeout=15).stdout.strip()
    except Exception:
        pass

    def read_json(path):
        try:
            return json.loads(Path(path).read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return None

    mono_check = None
    mono_ref_path = ROOT / 'reference_results' / 'single_depth_check.prediction.npy'
    mono_gpu_path = ROOT / 'outputs' / 'mono_gpu' / 'depthnet_z_m.npy'
    if mono_ref_path.is_file() and mono_gpu_path.is_file():
        mono_ref = load_depth(mono_ref_path)
        mono_gpu = load_depth(mono_gpu_path)
        if mono_ref.shape == mono_gpu.shape:
            both_m = np.isfinite(mono_ref) & np.isfinite(mono_gpu)
            md = np.abs(mono_gpu[both_m] - mono_ref[both_m])
            mono_check = {
                'reference': str(mono_ref_path.relative_to(ROOT)),
                'gpu_prediction': str(mono_gpu_path.relative_to(ROOT)),
                'comparable_pixels': int(both_m.sum()),
                'max_abs_diff_m': float(md.max()) if md.size else None,
                'mean_abs_diff_m': float(md.mean()) if md.size else None,
                'bitwise_equal': bool(np.array_equal(mono_gpu, mono_ref)),
            }

    manifest = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'verify_manifest.py')],
                              text=True, capture_output=True, cwd=str(ROOT))
    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    results = {
        'run_utc': started.isoformat(),
        'kit_root': str(ROOT),
        'hashes': {
            'rectified_left_png': sha256(left_path),
            'rectified_right_png': sha256(right_path),
            'ue_z_m_npy': sha256(gt_path),
            'model': sha256(ROOT / 'models' / 'yolo26s-depth.pt'),
            'depthnet_z_m_npy': sha256(OUT / 'depthnet_z_m.npy'),
            'sgbm_z_m_npy': sha256(OUT / 'sgbm_z_m.npy'),
        },
        'manifest': {'returncode': manifest.returncode,
                     'stdout': manifest.stdout.strip(),
                     'stderr': manifest.stderr.strip()},
        'cross_checks': {
            'environment': read_json(ROOT / 'outputs' / 'environment.json'),
            'replay_summary': read_json(ROOT / 'outputs' / 'replay_windows' / 'summary.json'),
            'mono_gpu_summary': read_json(ROOT / 'outputs' / 'mono_gpu' / 'summary.json'),
            'mono_gpu_vs_reference': mono_check,
        },
        'protocol': {'main_range_m': [0.0, 20.0], 'extended_range_m': [0.5, 80.0],
                     'posthoc_scale_fit': False,
                     'depth_definition': 'optical Z, metres',
                     'common_pixel_rule': 'truth in range and positive finite prediction for every method'},
        'geometry': meta,
        'timings': timings,
        'environment': env,
        'inference': details,
        'main': {k: v for k, v in protocol.items() if k != 'masks'},
        'extended': {k: v for k, v in extended.items() if k != 'masks'},
        'bad_rates_3m': bad_rates,
        'thresholds': thresholds,
        'depth_bins': bm,
        'disparity': disp_stats,
        'epipolar': epi_stats,
        'sgbm_sensitivity': sensitivity,
        'regions': regions,
        'region_metrics': region_rows,
        'cpu_reference_comparison': cpu_compare,
        'figures': {k: str(v.relative_to(ROOT)) for k, v in figures.items()},
        'outputs': {
            'depthnet_z_m': str((OUT / 'depthnet_z_m.npy').relative_to(ROOT)),
            'sgbm_z_m': str((OUT / 'sgbm_z_m.npy').relative_to(ROOT)),
            'sgbm_disparity_px': str((OUT / 'sgbm_disparity_px.npy').relative_to(ROOT)),
        },
    }
    (OUT / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False),
                                      encoding='utf-8')

    # write auto tables for the report
    write_report_tables(results)
    print(json.dumps({k: results[k] for k in ('main', 'extended', 'disparity',
                                              'epipolar', 'cpu_reference_comparison')},
                     indent=2, allow_nan=False))
    print('Wrote', OUT, 'and figures in', FIG)


def write_report_tables(results):
    main = results['main']
    rows = []
    for name in main['pairwise']:
        p = main['pairwise'][name]
        c = main['common_pixel_metrics'][name]
        rows.append([name, f'{p["pixels"]:,}', f'{p["coverage_of_gt"]*100:.2f}%',
                     f'{p["mae_m"]:.3f}', f'{p["rmse_m"]:.3f}',
                     f'{p["bias_m"]:+.3f}', f'{c["mae_m"]:.3f}',
                     f'{c["rmse_m"]:.3f}', f'{c["bias_m"]:+.3f}',
                     f'{c["median_abs_error_m"]:.3f}', f'{c["p90_abs_error_m"]:.3f}'])
    (TABLES / 'main_metrics.md').write_text(md_table(
        ['Method', 'Valid px', 'Coverage', 'MAE pairwise (m)', 'RMSE pairwise (m)',
         'Bias pairwise (m)', 'MAE common (m)', 'RMSE common (m)', 'Bias common (m)',
         'Median abs(err) common (m)', 'P90 abs(err) common (m)'],
        rows), encoding='utf-8')

    rows = []
    for r in results['depth_bins']:
        rows.append([r['method'], f'{r["bin_lo_m"]:g}–{r["bin_hi_m"]:g}',
                     f'{r["common_pixels"]:,}', f'{r["coverage_of_gt"]*100:.1f}%',
                     fmt(r['mae_m']), fmt(r['rmse_m']), fmt(r['bias_m'], signed=True),
                     fmt(r['p90_abs_error_m'])])
    (TABLES / 'depth_bins.md').write_text(md_table(
        ['Method', 'Truth bin (m)', 'Common px', 'Coverage', 'MAE (m)', 'RMSE (m)',
         'Bias (m)', 'P90 abs(err) (m)'], rows), encoding='utf-8')

    rows = []
    for name, vals in results['thresholds'].items():
        rows.append([name] + [f'{vals[k]*100:.1f}%' for k in
                              ('0.5m', '1m', '2m', '3m', '5m', '10m', '20m')])
    (TABLES / 'thresholds.md').write_text(md_table(
        ['Method', 'abs(err) ≥ 0.5 m', '≥ 1 m', '≥ 2 m', '≥ 3 m', '≥ 5 m',
         '≥ 10 m', '≥ 20 m'], rows), encoding='utf-8')

    rows = []
    for s in results['sgbm_sensitivity']:
        rows.append([s['block_size'], s['num_disparities'], s['uniqueness'],
                     f'{s["valid_pixels"]:,}', f'{s["coverage_of_gt"]*100:.1f}%',
                     fmt(s['mae_m']), fmt(s['rmse_m']), fmt(s['bias_m'], signed=True),
                     f'{s["runtime_ms"]:.0f}'])
    (TABLES / 'sgbm_sensitivity.md').write_text(md_table(
        ['Block size', 'Num disp', 'Uniqueness', 'Valid px', 'Coverage', 'MAE (m)',
         'RMSE (m)', 'Bias (m)', 'Runtime (ms)'], rows), encoding='utf-8')

    rows = []
    for r in results['region_metrics']:
        if r['common_pixels'] < 10:
            continue
        rows.append([r['region_id'], r['method'], fmt(r['gt_median_m'], 2),
                     f'{r["common_pixels"]:,}', fmt(r['mae_m']), fmt(r['rmse_m']),
                     fmt(r['bias_m'], signed=True)])
    (TABLES / 'cone_regions.md').write_text(md_table(
        ['Region', 'Method', 'GT median (m)', 'Common px', 'MAE (m)', 'RMSE (m)',
         'Bias (m)'], rows), encoding='utf-8')


if __name__ == '__main__':
    main()

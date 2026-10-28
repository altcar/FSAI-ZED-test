import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
prep = ROOT / 'outputs' / 'stereo_validation' / 'prepared'
L = cv2.imread(str(prep / 'left.png'), cv2.IMREAD_GRAYSCALE)
R = cv2.imread(str(prep / 'right.png'), cv2.IMREAD_GRAYSCALE)
gt = np.load(prep / 'ue_z_m.npy').astype(np.float32)
gvalid = np.isfinite(gt) & (gt > 0) & (gt <= 20)


def matcher():
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=160, blockSize=5,
        P1=8 * 25, P2=32 * 25, disp12MaxDiff=1, preFilterCap=63,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


dl = matcher().compute(L, R).astype(np.float32) / 16.0
dr = matcher().compute(R, L).astype(np.float32) / 16.0
print('dl valid >0:', int((dl > 0).sum()))
print('dr raw stats: >1', int((dr > 1).sum()), '<-1', int((dr < -1).sum()),
      'between', int(((dr >= -1) & (dr <= 1)).sum()))
print('dr q pos', np.percentile(dr[dr > 1], [1, 25, 50, 75, 99]))
if (dr < -1).any():
    print('dr q neg', np.percentile(dr[dr < -1], [1, 25, 50, 75, 99]))

h, w = dl.shape
ys, xs = np.nonzero(dl > 0)
d = dl[ys, xs]
for sign, name in [(1, 'dr as-is'), (-1, 'dr negated')]:
    drs = sign * dr
    for direction, dname in [(-1, 'x-d'), (1, 'x+d')]:
        xr = np.round(xs + direction * d).astype(np.int32)
        ok = (xr >= 0) & (xr < w)
        vals = np.full_like(d, np.nan)
        vals[ok] = drs[ys[ok], xr[ok]]
        for tol in (0.5, 1.0):
            pass_ = ok & np.isfinite(vals) & (np.abs(vals - d) <= tol)
            print(f'{name:12s} {dname:3s} tol={tol}: {pass_.sum():7d} '
                  f'({pass_.mean()*100:5.2f}% of left-valid)')

# Same-coordinate and shifted comparisons
for shift in (-2, -1, 0, 1, 2):
    xs2 = np.round(xs + shift).astype(np.int32)
    ok = (xs2 >= 0) & (xs2 < w)
    vals = np.full_like(d, np.nan)
    vals[ok] = dr[ys[ok], xs2[ok]]
    for tol in (0.5, 1.0):
        pass_ = ok & np.isfinite(vals) & (np.abs(vals - d) <= tol)
        print(f'dr as-is same+{shift:+d} tol={tol}: {pass_.sum():7d} '
              f'({pass_.mean()*100:5.2f}%)')

# Where does each left disparity value appear in the right row?
# For a few left-valid pixels, search the right row for a value within 0.5 px.
rng = np.random.default_rng(0)
sel = rng.choice(np.nonzero(dl > 0)[0].size, size=15, replace=False)
for idx in sel:
    y, x = ys[idx], xs[idx]
    target = d[idx]
    row = dr[y]
    hits = np.nonzero(np.abs(row - target) <= 0.5)[0]
    pred = x - target
    print(f'left ({x},{y}) d={target:.2f} -> expected right x={pred:.1f}; '
          f'hits={hits[:6].tolist()}{"..." if hits.size > 6 else ""}')

def eval_disp(dmap, label):
    z = np.full(dmap.shape, np.nan, np.float32)
    ok = dmap > 0
    z[ok] = 636.25 * 0.11972448013253958 / dmap[ok]
    m = gvalid & np.isfinite(z) & (z > 0) & (z <= 80)
    err = z[m] - gt[m]
    print(f'{label}: valid={int(ok.sum())} eval={int(m.sum())} '
          f'coverage={m.sum()/gvalid.sum()*100:.2f}% '
          f'MAE={np.abs(err).mean():.3f} RMSE={np.sqrt((err**2).mean()):.3f} '
          f'bias={err.mean():+.3f}')

eval_disp(dl, 'left disp (disp12MaxDiff=1)')
m2 = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=160, blockSize=5,
    P1=8 * 25, P2=32 * 25, disp12MaxDiff=-1, preFilterCap=63,
    uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
eval_disp(m2.compute(L, R).astype(np.float32) / 16.0, 'left disp (no internal L/R check)')

# Restrict to GT-valid left pixels to see the same thing where truth exists
g = np.full_like(dl, np.nan, np.float32)
gvalid = np.isfinite(gt) & (gt > 0) & (gt <= 20)
g[gvalid] = 636.25 * 0.11972448013253958 / gt[gvalid]
for sign, name in [(1, 'dr as-is'), (-1, 'dr negated')]:
    drs = sign * dr
    for direction, dname in [(-1, 'x-d'), (1, 'x+d')]:
        xr = np.round(xs + direction * d).astype(np.int32)
        ok = (xr >= 0) & (xr < w)
        vals = np.full_like(d, np.nan)
        vals[ok] = drs[ys[ok], xr[ok]]
        pass_ = ok & np.isfinite(vals) & gvalid[ys, xs] & (np.abs(vals - d) <= 1.0)
        print(f'GT subset {name:12s} {dname:3s}: {pass_.sum():7d} / {int((gvalid[ys,xs] & (dl[ys,xs]>0)).sum())}')

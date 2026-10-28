import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_stereo import K_TARGET

prep = ROOT / 'outputs' / 'stereo_validation' / 'prepared'
left = cv2.imread(str(prep / 'left.png'), cv2.IMREAD_GRAYSCALE)
right = cv2.imread(str(prep / 'right.png'), cv2.IMREAD_GRAYSCALE)
gt = np.load(prep / 'ue_z_m.npy').astype(np.float32)
f = float(K_TARGET[0, 0])
B = 0.11972448013253958

print('image', left.shape, 'gt', gt.shape)
print('left/right mean', left.mean(), right.mean(), 'std', left.std(), right.std())
print('abs diff mean', np.abs(left.astype(float) - right.astype(float)).mean())

expected = np.full(gt.shape, np.nan, np.float32)
valid_gt = np.isfinite(gt) & (gt > 0) & (gt <= 20)
expected[valid_gt] = f * B / gt[valid_gt]
print('expected disparity quantiles (GT 0-20):',
      np.percentile(expected[valid_gt], [0, 1, 5, 25, 50, 75, 95, 99, 100]))

for bs in (3, 5, 7, 9):
    for uniq in (5, 10, 15):
        m = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=160, blockSize=bs,
            P1=8 * bs * bs, P2=32 * bs * bs, disp12MaxDiff=1,
            preFilterCap=63, uniquenessRatio=uniq,
            speckleWindowSize=100, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        d = m.compute(left, right).astype(np.float32) / 16.0
        pos = d[d > 0]
        print(f'bs={bs} uniq={uniq}: valid={pos.size} ({pos.size/d.size:.2%})',
              'q=', np.percentile(pos, [0, 1, 25, 50, 75, 99, 100]).round(2)
              if pos.size else None)
        if bs == 5 and uniq == 10:
            d5 = d
            for dmax in (160, 256):
                m2 = cv2.StereoSGBM_create(
                    minDisparity=0, numDisparities=dmax, blockSize=bs,
                    P1=8 * bs * bs, P2=32 * bs * bs, disp12MaxDiff=-1,
                    preFilterCap=63, uniquenessRatio=0,
                    speckleWindowSize=0, speckleRange=0,
                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
                d2 = m2.compute(left, right).astype(np.float32) / 16.0
                pos2 = d2[d2 > 0]
                print(f'  loose dmax={dmax}: valid={pos2.size} ({pos2.size/d2.size:.2%})',
                      'q=', np.percentile(pos2, [0, 1, 25, 50, 75, 99, 100]).round(2)
                      if pos2.size else None)
                # error against expected on GT-valid pixels
                mm = valid_gt & (d2 > 0)
                if mm.any():
                    err = d2[mm] - expected[mm]
                    print('    vs expected: n', mm.sum(), 'median', np.median(err),
                          'mae', np.mean(np.abs(err)), 'within1', np.mean(np.abs(err) <= 1))
            break
    # no break

# Print a horizontal slice through a near cone, and expected disparity there
for y in (300, 360, 420, 470, 510, 540):
    row = d5[y]
    print('row', y, 'nonzero', int((row > 0).sum()),
          'gt median', np.nanmedian(gt[y, np.isfinite(gt[y])]),
          'expected median', np.nanmedian(expected[y, np.isfinite(expected[y])]),
          'disp median', np.median(row[row > 0]) if (row > 0).any() else None)

# Right matcher sign check
right_matcher = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=160, blockSize=5,
    P1=8 * 25, P2=32 * 25, disp12MaxDiff=1, preFilterCap=63,
    uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
dr = right_matcher.compute(right, left).astype(np.float32) / 16.0
print('right disp pos/neg counts', int((dr > 0).sum()), int((dr < 0).sum()),
      'median pos', np.median(dr[dr > 0]) if (dr > 0).any() else None,
      'median neg', np.median(dr[dr < 0]) if (dr < 0).any() else None)

import json
import numpy as np
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'outputs' / 'sdk_three_way'
z = np.load(out / 'sdk_z_m.npy', allow_pickle=False)
src = cv2.imread(str(ROOT / 'outputs/stereo_validation/prepared/left.png'))
sdk = cv2.imread(str(out / 'sdk_left.png'))
print('sdk depth finite_positive', int(np.isfinite(z).sum()), int((np.isfinite(z) & (z > 0)).sum()),
      'nan', int(np.isnan(z).sum()), 'min', float(np.nanmin(z)), 'max', float(np.nanmax(z)))
if src is not None and sdk is not None and src.shape == sdk.shape:
    diff = np.abs(src.astype(int) - sdk.astype(int))
    mse = (diff.astype(np.float64) ** 2).mean()
    print('sdk_left vs prepared_left: mean_abs', diff.mean(), 'max', diff.max(),
          'psnr', 10 * np.log10(255 ** 2 / max(mse, 1e-9)))
summary = json.loads((out / 'comparison' / 'summary.json').read_text(encoding='utf-8'))
print('summary sdk pixels', summary['pairwise']['ZED_SDK']['pixels'],
      'coverage', summary['pairwise']['ZED_SDK']['coverage_of_gt'],
      'common', summary['common_pixels'], 'three_way_complete', summary['three_way_complete'])
(out / 'INVALID_CAPTURE.txt').write_text(
    'INVALID SDK CAPTURE: sdk_left.png does not match the prepared left image and '
    'sdk_z_m.npy contains no valid depth. Do not use the comparison/summary.json '
    'metrics as a ZED SDK or three-way result. See ZED_SDK_SIMULATION_RESULT.md.\n',
    encoding='utf-8')

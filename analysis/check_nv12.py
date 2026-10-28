import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
bgr = cv2.imread(str(ROOT / 'outputs/stereo_validation/prepared/left.png'))
h, w = bgr.shape[:2]


def bgr_to_nv12(bgr):
    h, w = bgr.shape[:2]
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    y = i420[:h, :]
    u = i420[h:h + h // 4, :].reshape(h // 2, w // 2)
    v = i420[h + h // 4:h + h // 2, :].reshape(h // 2, w // 2)
    uv = np.empty((h // 2, w), np.uint8)
    uv[:, 0::2] = u
    uv[:, 1::2] = v
    return np.vstack([y, uv])


nv12 = bgr_to_nv12(bgr)
back = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
diff = np.abs(back.astype(int) - bgr.astype(int))
print('shapes', nv12.shape, back.shape)
print('mean_abs_diff', diff.mean(), 'max', diff.max())
mse = (diff.astype(np.float64) ** 2).mean()
print('psnr', 10 * np.log10(255 ** 2 / max(mse, 1e-9)))
cv2.imwrite(str(ROOT / 'outputs/logs/nv12_roundtrip.png'), np.hstack([bgr, back]))

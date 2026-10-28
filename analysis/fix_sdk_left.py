"""Fix the BGRA/BGR channel swap in the SDK-saved left image.

The SDK's VIEW.LEFT is written through cv2.imwrite as a 4-channel image; the
saved file has R/B swapped relative to the prepared left.  The depth itself does
not care, but DepthNet must see the correct channel order.
"""
from PIL import Image
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / 'outputs/sdk_three_way/sdk_left.png'
out = ROOT / 'outputs/sdk_three_way/sdk_left_corrected.png'
arr = np.array(Image.open(src).convert('RGB'))
corrected = arr[:, :, ::-1]  # swap R and B
Image.fromarray(corrected).save(out)
prep = np.array(Image.open(ROOT / 'outputs/stereo_validation/prepared/left.png').convert('RGB')).astype(float)
for name, a in [('as-is', arr.astype(float)), ('corrected', corrected.astype(float))]:
    mse = ((prep - a) ** 2).mean()
    print(name, 'PSNR', 10 * np.log10(255 ** 2 / max(mse, 1e-9)))
print('wrote', out)

"""Rectify the UE pair into a parallel virtual GS 4mm rig, preserving metric Z."""
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIZE = (960, 600)
K_TARGET = np.array([[636.25, 0, 480], [0, 636.25, 300], [0, 0, 1]], dtype=np.float64)


def rotate_depth(depth, source_k, rect_rotation, target_k, size=SIZE):
    """Inverse ray mapping for a pure camera rotation, with optical-Z correction.

    Nearest-neighbour depth avoids interpolating foreground/background depths.
    At boundaries, a one-pixel sampling uncertainty remains.
    """
    u, v = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    rays = np.linalg.inv(target_k) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    source_rays = rect_rotation.T @ rays
    pixels = source_k @ source_rays
    x = (pixels[0] / pixels[2]).reshape(v.shape).astype(np.float32)
    y = (pixels[1] / pixels[2]).reshape(v.shape).astype(np.float32)
    zfactor = source_rays[2].reshape(v.shape)
    sampled = cv2.remap(depth.astype(np.float32), x, y, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=float('nan'))
    with np.errstate(invalid='ignore', divide='ignore'):
        out = sampled / zfactor
    out[(zfactor <= 0) | ~np.isfinite(out) | (out <= 0)] = np.nan
    return out.astype(np.float32)


def prepare(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    c = json.loads((source / 'calibration.json').read_text(encoding='utf-8'))
    kl, kr = [np.array(c['eyes'][eye]['K'], dtype=np.float64) for eye in ('left', 'right')]
    dl, dr = [np.array(c['eyes'][eye]['D'], dtype=np.float64) for eye in ('left', 'right')]
    rotation = np.array(c['R_right_from_left'], dtype=np.float64)
    translation = np.array(c['t_right_from_left_m'], dtype=np.float64).reshape(3, 1)
    r1, r2, *_ = cv2.stereoRectify(kl, dl, kr, dr, (c['width'], c['height']), rotation,
                                  translation, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    valid_masks = []
    for eye, k, d, r in [('left', kl, dl, r1), ('right', kr, dr, r2)]:
        raw = cv2.imread(str(source / f'{eye}_rgb_raw.png'), cv2.IMREAD_COLOR)
        if raw is None:
            raise FileNotFoundError(eye + '_rgb_raw.png')
        mx, my = cv2.initUndistortRectifyMap(k, d, r, K_TARGET, SIZE, cv2.CV_32FC1)
        valid_masks.append((mx >= 1) & (mx < raw.shape[1] - 2) & (my >= 1) & (my < raw.shape[0] - 2))
        rect = cv2.remap(raw, mx, my, cv2.INTER_LINEAR)
        if not cv2.imwrite(str(output / f'{eye}.png'), rect):
            raise OSError('Failed to write rectified PNG')
    gt = rotate_depth(np.load(source / 'left_depth_z_pinhole_m.npy', allow_pickle=False), kl, r1, K_TARGET)
    # Keep truth only where both source images cover the output; no invented border pixels.
    gt[~(valid_masks[0] & valid_masks[1])] = np.nan
    np.save(output / 'ue_z_m.npy', gt)
    meta = {'width': SIZE[0], 'height': SIZE[1], 'K': K_TARGET.tolist(),
            'R_rect_from_original_left': r1.tolist(), 'baseline_m': float(np.linalg.norm(translation)),
            'model_id': 30, 'sim_lens_type': 1, 'profile': 'virtual ZED X One GS narrow, parallel stereo',
            'source': 'data/stereo_demo, original ZED SN2448592 UE render',
            'depth_definition': 'optical_z_metres_in_rectified_left',
            'truth_resampling': 'nearest depth, inverse rotated rays, corrected optical Z',
            'both_eye_image_coverage_fraction': float((valid_masks[0] & valid_masks[1]).mean())}
    (output / 'geometry.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta


if __name__ == '__main__':
    print(json.dumps(prepare(ROOT / 'data/stereo_demo', ROOT / 'outputs/stereo_prepared'), indent=2))

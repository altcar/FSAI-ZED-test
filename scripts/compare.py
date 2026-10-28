"""Run the bundled FSAI-Seg depth model and compare metric optical Z, in metres."""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor'))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'outputs' / 'yolo_config'))
import numpy as np


def load_depth(path):
    a = np.load(path, allow_pickle=False)
    if a.ndim != 2:
        raise ValueError(f'{path}: expected HxW metric depth, got {a.shape}')
    return a.astype(np.float32)


def metrics(gt, pred, mask):
    count = int(mask.sum())
    if not count:
        return {'pixels': 0, 'mae_m': None, 'rmse_m': None, 'bias_m': None, 'abs_rel': None}
    delta = pred[mask].astype(np.float64) - gt[mask].astype(np.float64)
    return {'pixels': count, 'mae_m': float(np.abs(delta).mean()),
            'rmse_m': float(np.sqrt(np.mean(delta ** 2))), 'bias_m': float(delta.mean()),
            'abs_rel': float(np.mean(np.abs(delta) / gt[mask]))}


def evaluate(gt, predictions, minimum=0.0, maximum=20.0):
    target = np.isfinite(gt) & (gt > minimum) & (gt <= maximum)
    if not target.any():
        raise ValueError('No UE truth pixels in evaluation range')
    masks = {}
    common = target.copy()
    for name, pred in predictions.items():
        if pred.shape != gt.shape:
            raise ValueError(f'{name}: {pred.shape} != truth {gt.shape}; geometric alignment is required')
        masks[name] = target & np.isfinite(pred) & (pred > 0)
        common &= masks[name]
    report = {'depth_definition': 'optical_z_metres', 'gt_min_exclusive_m': minimum,
              'gt_max_inclusive_m': maximum, 'target_pixels': int(target.sum()),
              'posthoc_scale_fit': False, 'pairwise': {}, 'common_pixels': int(common.sum()),
              'common_pixel_metrics': {}}
    for name, pred in predictions.items():
        report['pairwise'][name] = metrics(gt, pred, masks[name])
        report['pairwise'][name]['coverage_of_gt'] = float(masks[name].sum() / target.sum())
        report['common_pixel_metrics'][name] = metrics(gt, pred, common)
    return report, target, masks, common


def infer(image, device):
    from ultralytics import YOLO
    import torch
    if device == 'auto':
        device = '0' if torch.cuda.is_available() else 'cpu'
    if device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install CUDA PyTorch, or use --device cpu')
    model = YOLO(str(ROOT / 'models' / 'yolo26s-depth.pt'))
    result = model.predict(source=str(image), device=device, imgsz=768, verbose=True)[0]
    if result.depth is None:
        raise RuntimeError('Model did not return depth; use the bundled ultralytics source')
    depth = result.depth.data.detach().cpu().numpy().squeeze()
    return depth.astype(np.float32), {'device': device, 'torch': torch.__version__,
                                     'ultralytics': __import__('ultralytics').__version__,
                                     'imgsz': 768, 'speed_ms': result.speed}


def plot(image, gt, predictions, masks, common, maximum, output):
    import matplotlib.pyplot as plt
    from PIL import Image
    n = len(predictions)
    fig, axes = plt.subplots(2, n + 1, figsize=(5 * (n + 1), 8), squeeze=False,
                             layout='constrained')
    axes[0, 0].imshow(Image.open(image))
    axes[0, 0].set_title('Exact input left image')
    valid_gt = np.isfinite(gt) & (gt > 0) & (gt <= maximum)
    cm = plt.get_cmap('viridis').copy()
    cm.set_bad('#bbbbbb')
    im = axes[1, 0].imshow(np.ma.array(gt, mask=~valid_gt), vmin=0, vmax=maximum, cmap=cm)
    axes[1, 0].set_title('UE optical Z (m)')
    for column, (name, pred) in enumerate(predictions.items(), 1):
        axes[0, column].imshow(np.ma.array(pred, mask=~masks[name]), vmin=0, vmax=maximum, cmap=cm)
        axes[0, column].set_title(name + ' (m)')
        err = np.abs(pred - gt)
        err_image = axes[1, column].imshow(np.ma.array(err, mask=~common), vmin=0, vmax=5, cmap='magma')
        axes[1, column].set_title('|prediction - UE|, common pixels, 0..5 m')
    fig.colorbar(im, ax=axes[0, :].tolist() + [axes[1, 0]], label='Depth (m)', shrink=0.65)
    fig.colorbar(err_image, ax=axes[1, 1:].tolist(), label='Absolute error (m)', shrink=0.65)
    for ax in axes.flat:
        ax.axis('off')
    fig.savefig(output, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, default=ROOT / 'data/mono_demo/image_pinhole.png')
    p.add_argument('--gt', type=Path, default=ROOT / 'data/mono_demo/depth_z_pinhole_m.npy')
    p.add_argument('--prediction', type=Path, help='Replay an existing native metric prediction')
    p.add_argument('--sdk-depth', type=Path, help='SDK MEASURE.DEPTH in metres, registered to --image')
    p.add_argument('--instances', type=Path, help='Optional aligned uint16 instance PNG; 0 is background')
    p.add_argument('--device', default='auto')
    p.add_argument('--min-depth', type=float, default=0)
    p.add_argument('--max-depth', type=float, default=20)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/mono')
    args = p.parse_args()
    if not (0 <= args.min_depth < args.max_depth < float('inf')):
        p.error('Require 0 <= min-depth < max-depth < infinity')
    from PIL import Image
    gt = load_depth(args.gt)
    with Image.open(args.image) as im:
        if (im.height, im.width) != gt.shape:
            raise ValueError('Image and truth dimensions differ')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prediction:
        pred, details = load_depth(args.prediction), {'replayed_prediction': str(args.prediction)}
    else:
        pred, details = infer(args.image, args.device)
    predictions = {'DepthNet': pred}
    if args.sdk_depth:
        predictions['ZED_SDK'] = load_depth(args.sdk_depth)
    report, target, masks, common = evaluate(gt, predictions, args.min_depth, args.max_depth)
    report['three_way_complete'] = len(predictions) == 2
    report['inference'] = details
    report['image_sha256'] = hashlib.sha256(args.image.read_bytes()).hexdigest()
    report['gt_sha256'] = hashlib.sha256(args.gt.read_bytes()).hexdigest()
    np.save(args.output / 'depthnet_z_m.npy', pred)
    np.save(args.output / 'common_valid.npy', common)
    for name, a in predictions.items():
        e = np.full(gt.shape, np.nan, dtype=np.float32)
        e[masks[name]] = np.abs(a[masks[name]] - gt[masks[name]])
        np.save(args.output / f'{name}_absolute_error_m.npy', e)
    if args.instances:
        ids = np.asarray(Image.open(args.instances))
        if ids.ndim != 2 or ids.shape != gt.shape:
            raise ValueError('Instance PNG must be aligned HxW IDs')
        rows = []
        cone_common = common & (ids != 0)
        report['all_cone_pixels_common'] = {n: metrics(gt, a, cone_common) for n, a in predictions.items()}
        for ident in np.unique(ids):
            if ident == 0:
                continue
            region = common & (ids == ident)
            for name, a in predictions.items():
                rows.append({'instance_id': int(ident), 'method': name, **metrics(gt, a, region)})
        if rows:
            with (args.output / 'cones.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    plot(args.image, gt, predictions, masks, common, args.max_depth, args.output / 'comparison.png')
    print(json.dumps(report, indent=2, allow_nan=False))
    print('Results:', args.output.resolve())


if __name__ == '__main__':
    main()

"""Experimental end-to-end UE stereo -> SDK -> exact SDK left image -> DepthNet.

Requires Windows + NVIDIA + ZED SDK 5.5.x + its pyzed + matching sim C exports.
Refuses unsupported calibration instead of reporting a misleading depth comparison.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import cv2
import numpy as np
from prepare_stereo import ROOT, SIZE, K_TARGET, prepare, rotate_depth


def calibration_text(baseline):
    lines = ['[STEREO]', f'Baseline={baseline * 1000:.9f}', 'TY=0', 'TZ=0']
    for resolution in ('FHD', 'SVGA', 'FHD1200', 'QHDPLUS', '4K'):
        lines += [f'CV_{resolution}=0', f'RX_{resolution}=0', f'RZ_{resolution}=0']
    lines += ['', '[SIM]', 'lens_type=1', 'camera_model=30', '']
    return '\n'.join(lines)


def camera_k(camera):
    return np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]], dtype=float)


def find_dll(explicit):
    if explicit:
        return explicit.resolve()
    candidates = [ROOT / 'zed_runtime/sl_zed64.dll']
    sdk = Path(os.environ.get('ZED_SDK_ROOT_DIR', r'C:\Program Files (x86)\ZED SDK'))
    candidates += [sdk / 'bin/sl_zed64.dll', sdk / 'bin/sl_zed.dll']
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('No sim runtime DLL. Put compatible sl_zed64.dll in zed_runtime, or pass --dll')


def capture(args, prepared, meta):
    import pyzed.sl as sl
    if tuple(map(int, sl.Camera.get_sdk_version().split('.')[:2])) != (5, 5):
        raise RuntimeError('Use SDK/pyzed 5.5.x for this experimental sim ABI')
    zed = sl.Camera()
    init = sl.InitParameters()
    init.set_from_stream('127.0.0.1', args.port)
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode)
    init.depth_stabilization = 0
    init.camera_disable_self_calib = True
    init.depth_minimum_distance = 0.3
    init.depth_maximum_distance = 80.0
    init.optional_settings_path = str(prepared / 'settings') + os.sep
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        zed.close()
        raise RuntimeError(f'SDK open failed: {status}')
    try:
        info = zed.get_camera_information()
        cfg = info.camera_configuration
        raw, rect = cfg.calibration_parameters_raw, cfg.calibration_parameters
        if (cfg.resolution.width, cfg.resolution.height) != SIZE:
            raise RuntimeError('Unexpected SDK image resolution; no comparison produced')
        for eye in ('left_cam', 'right_cam'):
            cam = getattr(raw, eye)
            if not np.allclose(camera_k(cam), K_TARGET, atol=0.75, rtol=0):
                raise RuntimeError(f'SDK raw intrinsics do not match prepared pixels: {eye}, {camera_k(cam)}')
            if np.max(np.abs(np.asarray(cam.disto))) > 1e-5:
                raise RuntimeError('SDK expects distorted inputs; prepared inputs are pinhole')
        raw_rotation = np.asarray(raw.stereo_transform.get_rotation_matrix().r).reshape(3, 3)
        raw_t = np.asarray(raw.stereo_transform.get_translation().get()).reshape(3)
        if not np.allclose(raw_rotation, np.eye(3), atol=1e-4):
            raise RuntimeError('SDK changed stereo rotation; refusing unregistered truth')
        # SDK releases may expose the right-eye pose (+B) rather than the
        # OpenCV left-to-right transform (-B); this parallel rig needs |Tx|=B.
        if not np.allclose(np.abs(raw_t), [meta['baseline_m'], 0, 0], atol=0.0002):
            raise RuntimeError(f'SDK baseline mismatch (metres): {raw_t}')
        runtime = sl.RuntimeParameters()
        runtime.confidence_threshold = 95
        runtime.texture_confidence_threshold = 100
        rgb, depth = sl.Mat(), sl.Mat()
        successful = 0
        timeout = time.monotonic() + 180
        while successful < args.warmup and time.monotonic() < timeout:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                successful += 1
        if successful < args.warmup:
            raise RuntimeError('Timed out waiting for SDK frames')
        # These two arrays come from the same last successful grab.
        for status in [zed.retrieve_image(rgb, sl.VIEW.LEFT), zed.retrieve_measure(depth, sl.MEASURE.DEPTH)]:
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f'SDK retrieval failed: {status}')
        left = rgb.get_data().copy()
        z = depth.get_data().copy().astype(np.float32).squeeze()
        if z.shape != (SIZE[1], SIZE[0]) or left.shape[:2] != z.shape:
            raise RuntimeError('SDK output dimensions differ')
        if not np.any(np.isfinite(z) & (z > 0)):
            raise RuntimeError('SDK produced no valid depth')
        # Use SDK rectified K, after checking it consumed the intended parallel rig.
        k = camera_k(rect.left_cam)
        source = ROOT / 'data/stereo_demo'
        c = json.loads((source / 'calibration.json').read_text(encoding='utf-8'))
        gt = rotate_depth(np.load(source / 'left_depth_z_pinhole_m.npy', allow_pickle=False),
                          np.array(c['eyes']['left']['K']),
                          np.array(meta['R_rect_from_original_left']), k)
        # Also restrict to coverage of the prepared images, reprojected into SDK K.
        coverage = np.isfinite(np.load(prepared / 'ue_z_m.npy')).astype(np.uint8)
        mx, my = cv2.initUndistortRectifyMap(K_TARGET, np.zeros(5), np.eye(3), k, SIZE, cv2.CV_32FC1)
        valid = cv2.remap(coverage, mx, my, cv2.INTER_NEAREST) > 0
        gt[~valid] = np.nan
        cv2.imwrite(str(args.output / 'sdk_left.png'), left)
        np.save(args.output / 'sdk_z_m.npy', z)
        np.save(args.output / 'ue_z_m.npy', gt)
        metadata = {'sdk_version': sl.Camera.get_sdk_version(), 'sdk_depth_mode': args.depth_mode,
                    'K_rectified': k.tolist(), 'K_raw': camera_k(raw.left_cam).tolist(),
                    'raw_translation_m': raw_t.tolist(), 'depth_unit': 'metres',
                    'image_timestamp_ns': int(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()),
                    'input': 'repeated static UE stereo pair; RGB only, no GT ingestion',
                    'self_calibration': False, 'depth_stabilization': 0,
                    'confidence_threshold': 95, 'texture_confidence_threshold': 100,
                    'virtual_geometry': meta,
                    'stream_codec': 'H264, 3000 kbps, chunk 1200, transport BOTH (lossy)',
                    'sim_runtime': 'sl_zed64.dll 5.5.1 (5.2.0-labelled)',
                    'metadata_note': 'producer metadata 1216 bytes; 5.5.1 consumer expects 2904; camera settings skipped'}
        (args.output / 'sdk_capture.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    finally:
        zed.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dll', type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/sdk_three_way')
    p.add_argument('--port', type=int, default=30000)
    p.add_argument('--depth-mode', choices=['NEURAL', 'NEURAL_LIGHT', 'PERFORMANCE', 'QUALITY', 'ULTRA'], default='NEURAL')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--device', default='0')
    args = p.parse_args()
    if os.name != 'nt':
        raise RuntimeError('Run this SDK bridge on the recipient Windows/NVIDIA computer')
    if args.port % 2 or not (1024 <= args.port <= 65534) or args.warmup < 1:
        p.error('Use an even port in 1024..65534 and warmup >=1')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # A unique run directory avoids stale ready markers and preserves previous captures.
    import tempfile
    prepared = Path(tempfile.mkdtemp(prefix='bridge_', dir=args.output))
    meta = prepare(ROOT / 'data/stereo_demo', prepared)
    dll = find_dll(args.dll)
    serial = 110307008
    command = [sys.executable, str(ROOT / 'scripts/sim_streamer_fixed.py'), '--dll', str(dll),
               '--prepared', str(prepared), '--serial', str(serial), '--port', str(args.port)]
    subprocess.run(command + ['--probe'], check=True)
    settings = prepared / 'settings'
    settings.mkdir()
    conf = calibration_text(meta['baseline_m'])
    (settings / f'SN{serial}.conf').write_text(conf, encoding='ascii')
    # init_streamer has no settings-path argument. Supply only a reserved virtual SN
    # in the SDK default folder, and delete our own newly created file on exit.
    default = Path(os.environ.get('PROGRAMDATA', r'C:\ProgramData')) / 'Stereolabs/settings'
    default.mkdir(parents=True, exist_ok=True)
    target = default / f'SN{serial}.conf'
    if target.exists():
        raise FileExistsError(f'{target} already exists; move it aside explicitly before this demo')
    created = False
    proc = None
    log = (args.output / 'streamer.log').open('w', encoding='utf-8')
    try:
        with target.open('x', encoding='ascii') as f:
            f.write(conf)
        created = True
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        timeout = time.monotonic() + 45
        while not (prepared / 'stream_ready.json').exists():
            if proc.poll() is not None:
                raise RuntimeError('Streamer exited. Read outputs/sdk_three_way/streamer.log')
            if time.monotonic() > timeout:
                raise RuntimeError('Streamer initialization timed out')
            time.sleep(0.1)
        capture(args, prepared, meta)
    finally:
        (prepared / 'STOP').touch()
        if proc is not None:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=10)
        log.close()
        if created and target.read_text(encoding='ascii') == conf:
            target.unlink()
    # The 04 wrapper runs fix_sdk_left.py + compare.py on the corrected image.


if __name__ == '__main__':
    main()

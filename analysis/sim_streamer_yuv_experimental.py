"""Experimental Windows wrapper for Stereolabs' sim-streamer C ABI (SDK >=5.5).

ABI source: licenses/types_c.h and zed_interface_loader.hpp, official repository
stereolabs/zed-isaac-sim at f9160973f321ec743f626188244cd9702d85bec6.
This process receives only PNGs. It never reads or streams UE truth.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np


class StreamingParameters(C.Structure):
    _fields_ = [('mode', C.c_int), ('imu_cam_q', C.c_float * 4), ('imu_cam_t', C.c_float * 3),
                ('image_width', C.c_int), ('image_height', C.c_int), ('codec_type', C.c_int),
                ('port', C.c_ushort), ('fps', C.c_int), ('serial_number', C.c_int),
                ('alpha_channel_included', C.c_bool), ('input_format', C.c_int),
                ('verbose', C.c_bool), ('transport_layer_mode', C.c_int), ('bitrate', C.c_int),
                ('chunk_size', C.c_ushort), ('gop_size', C.c_int), ('gpu_input', C.c_bool),
                ('stream_depth', C.c_bool), ('depth_width', C.c_int), ('depth_height', C.c_int),
                ('depth_bitrate', C.c_int)]


def load_library(path):
    if os.name != 'nt':
        raise RuntimeError('The sim-streamer bridge requires Windows/NVIDIA; not available on this Mac')
    path = Path(path).resolve()
    # Keep directory handles alive while the native library uses dependencies.
    handles = [os.add_dll_directory(str(path.parent))]
    for candidate in [os.environ.get('CUDA_PATH'), os.environ.get('ZED_SDK_ROOT_DIR')]:
        if candidate and (Path(candidate) / 'bin').is_dir():
            handles.append(os.add_dll_directory(str(Path(candidate) / 'bin')))
    lib = C.CDLL(str(path))
    try:
        get_version = lib.getZEDSDKRuntimeVersion_C
        get_version.argtypes = [C.POINTER(C.c_int)] * 3
        get_version.restype = C.c_int
        version = [C.c_int() for _ in range(3)]
        get_version(*(C.byref(x) for x in version))
        version = tuple(x.value for x in version)
        if version[:2] != (5, 5):
            raise RuntimeError(f'Unverified sim ABI {version}; this bridge targets SDK 5.5.x only')
        lib.init_streamer.argtypes = [C.c_int, C.POINTER(StreamingParameters)]
        lib.init_streamer.restype = C.c_int
        lib.stream_yuv.argtypes = [C.c_int, C.c_void_p, C.c_void_p, C.c_longlong] + [C.c_float] * 7
        lib.stream_yuv.restype = C.c_int
        lib.close_streamer.argtypes = [C.c_int]
        lib.close_streamer.restype = None
        lib.is_sn_valid.argtypes = [C.c_int]
        lib.is_sn_valid.restype = C.c_bool
    except AttributeError as e:
        raise RuntimeError('DLL lacks the official sim-streamer exports. Use a matching SDK 5.5 sim runtime; '
                           'ordinary pyzed installation alone may not supply these exports.') from e
    return lib, version, handles


def bgr_to_nv12(bgr):
    """BGR uint8 -> NV12 (Y plane, then interleaved UV), shape (h*3/2, w)."""
    h, w = bgr.shape[:2]
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    y = i420[:h, :]
    u = i420[h:h + h // 4, :].reshape(h // 2, w // 2)
    v = i420[h + h // 4:h + h // 2, :].reshape(h // 2, w // 2)
    uv = np.empty((h // 2, w), np.uint8)
    uv[:, 0::2] = u
    uv[:, 1::2] = v
    return np.vstack([y, uv])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dll', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--serial', type=int, default=110307008)
    p.add_argument('--port', type=int, default=30000)
    p.add_argument('--fps', type=int, default=15)
    p.add_argument('--probe', action='store_true')
    args = p.parse_args()
    lib, version, handles = load_library(args.dll)
    if not lib.is_sn_valid(args.serial):
        raise RuntimeError('Runtime rejected the virtual serial number')
    print('Sim runtime:', version, 'ABI structure bytes:', C.sizeof(StreamingParameters), flush=True)
    if args.probe:
        return
    images = []
    width = height = None
    for eye in ('left', 'right'):
        bgr = cv2.imread(str(args.prepared / f'{eye}.png'))
        if bgr is None:
            raise FileNotFoundError(eye)
        width, height = bgr.shape[1], bgr.shape[0]
        images.append(np.ascontiguousarray(bgr_to_nv12(bgr)))
    if images[0].shape != images[1].shape:
        raise ValueError('Stereo sizes differ')
    params = StreamingParameters()
    params.mode = 1
    params.imu_cam_q[:] = (0, 0, 0, 1)
    params.image_width, params.image_height = width, height
    params.codec_type, params.port, params.fps = 1, args.port, args.fps
    params.serial_number = args.serial
    params.alpha_channel_included, params.input_format = False, 2
    params.verbose, params.transport_layer_mode = True, 0
    params.bitrate, params.chunk_size, params.gop_size = 16000, 4096, -1
    # stream_depth remains false: the SDK must estimate depth from the stereo pair.
    status = lib.init_streamer(0, C.byref(params))
    if status != 0:
        raise RuntimeError(f'init_streamer failed: status {status}; see licenses/types_c.h')
    (args.prepared / 'stream_ready.json').write_text(json.dumps({'runtime': version}), encoding='utf-8')
    try:
        deadline = time.monotonic() + 1800
        frame = 0
        while not (args.prepared / 'STOP').exists() and time.monotonic() < deadline:
            start = time.monotonic()
            ts = frame * (1_000_000_000 // args.fps)
            status = lib.stream_yuv(0, images[0].ctypes.data, images[1].ctypes.data,
                                    ts, 1, 0, 0, 0, 0, 0, 0)
            frame += 1
            if status not in (0, -1):
                raise RuntimeError(f'stream_yuv failed: status {status}')
            time.sleep(max(0.001, 1 / args.fps - (time.monotonic() - start)))
    finally:
        lib.close_streamer(0)


if __name__ == '__main__':
    main()

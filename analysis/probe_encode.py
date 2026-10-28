"""Probe init_streamer + one stream_rgb call for one format combination."""
import argparse
import ctypes as C
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SDK = Path(os.environ.get(
    'FSAI_ZED_SDK_ROOT', r'L:\save\userdata\Downloads\ZED_SDK_extract\app'))
DLL = ROOT / 'zed_runtime' / 'sl_zed64.dll'
SERIAL = 110307008


class Params(C.Structure):
    _fields_ = [
        ('mode', C.c_int), ('imu_cam_q', C.c_float * 4), ('imu_cam_t', C.c_float * 3),
        ('image_width', C.c_int), ('image_height', C.c_int), ('codec_type', C.c_int),
        ('port', C.c_ushort), ('fps', C.c_int), ('serial_number', C.c_int),
        ('alpha_channel_included', C.c_bool), ('input_format', C.c_int),
        ('verbose', C.c_bool), ('transport_layer_mode', C.c_int), ('bitrate', C.c_int),
        ('chunk_size', C.c_ushort), ('gop_size', C.c_int),
        ('adaptative_bitrate', C.c_bool), ('enable_downscale', C.c_bool),
        ('pad_extra', C.c_byte * 2),
        ('target_width', C.c_int), ('target_height', C.c_int),
        ('target_bitrate', C.c_int),
        ('gpu_input', C.c_bool), ('stream_depth', C.c_bool),
        ('pad2', C.c_byte * 2),
        ('depth_width', C.c_int), ('depth_height', C.c_int),
        ('depth_bitrate', C.c_int),
        ('_tail', C.c_byte * 140),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--width', type=int, default=960)
    p.add_argument('--height', type=int, default=600)
    p.add_argument('--codec', type=int, default=1)
    p.add_argument('--fmt', type=int, default=0)
    p.add_argument('--alpha', type=int, default=1)
    p.add_argument('--channels', type=int, default=0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--mode', type=int, default=1)
    p.add_argument('--transport', type=int, default=0)
    p.add_argument('--bitrate', type=int, default=16000)
    p.add_argument('--chunk', type=int, default=1200)
    p.add_argument('--fps', type=int, default=15)
    p.add_argument('--frames', type=int, default=3)
    p.add_argument('--cuda-init', type=int, default=0)
    p.add_argument('--yuv', type=int, default=0)
    p.add_argument('--func', choices=['rgb', 'yuv'], default='rgb')
    p.add_argument('--yuvlayout', choices=['nv12', 'yuv422_row', 'yuv422_col', 'yuv444',
                                           'yuyv', 'yvyu', 'yv12', 'i420'],
                   default='nv12')
    p.add_argument('--ts', default='epoch')
    p.add_argument('--same', type=int, default=0)
    p.add_argument('--pair', type=int, default=0)
    p.add_argument('--port', type=int, default=30300)
    p.add_argument('--dll', type=Path, default=DLL)
    args = p.parse_args()

    handles = [os.add_dll_directory(str(args.dll.parent))]
    if (SDK / 'bin').is_dir():
        handles.append(os.add_dll_directory(str(SDK / 'bin')))
    lib = C.CDLL(str(args.dll))
    lib.getZEDSDKRuntimeVersion_C.argtypes = [C.POINTER(C.c_int)] * 3
    lib.getZEDSDKRuntimeVersion_C.restype = C.c_int
    v = [C.c_int() for _ in range(3)]
    lib.getZEDSDKRuntimeVersion_C(*(C.byref(x) for x in v))
    lib.is_sn_valid.argtypes = [C.c_int]
    lib.is_sn_valid.restype = C.c_bool
    if not lib.is_sn_valid(SERIAL):
        print('SN rejected'); return 2

    settings = Path(os.environ.get('PROGRAMDATA', r'C:\ProgramData')) / 'Stereolabs/settings'
    settings.mkdir(parents=True, exist_ok=True)
    conf = settings / f'SN{SERIAL}.conf'
    if not conf.exists():
        src = next((ROOT / 'outputs/sdk_three_way').glob('bridge_*/settings/*.conf'), None)
        if src:
            shutil.copyfile(src, conf)

    if args.cuda_init:
        import torch
        torch.cuda.init()
        torch.cuda.set_device(0)
        print('cuda device', torch.cuda.current_device(), flush=True)

    params = Params()
    params.mode = args.mode
    params.imu_cam_q[:] = (0, 0, 0, 1)
    params.image_width, params.image_height = args.width, args.height
    params.codec_type, params.port, params.fps = args.codec, args.port, args.fps
    params.serial_number = SERIAL
    params.alpha_channel_included = bool(args.alpha) and not (args.yuv or args.func == 'yuv')
    params.input_format = 2 if (args.yuv or args.func == 'yuv') else args.fmt
    params.verbose, params.transport_layer_mode = True, args.transport
    params.bitrate, params.chunk_size, params.gop_size = args.bitrate, args.chunk, -1
    params.adaptative_bitrate, params.enable_downscale = False, False
    params.target_width, params.target_height = args.width, args.height
    params.target_bitrate = args.bitrate
    params.gpu_input = bool(args.gpu)

    lib.init_streamer.argtypes = [C.c_int, C.POINTER(Params)]
    lib.init_streamer.restype = C.c_int
    if args.func == 'yuv':
        lib.stream_yuv.argtypes = [C.c_int, C.c_void_p, C.c_void_p, C.c_longlong] + [C.c_float] * 7
        lib.stream_yuv.restype = C.c_int
        stream_fn = lib.stream_yuv
    else:
        lib.stream_rgb.argtypes = [C.c_int, C.c_void_p, C.c_void_p, C.c_longlong] + [C.c_float] * 7
        lib.stream_rgb.restype = C.c_int
        stream_fn = lib.stream_rgb
    lib.close_streamer.argtypes = [C.c_int]

    status = lib.init_streamer(0, C.byref(params))
    print('init', status, flush=True)
    if status != 0:
        return 1

    h, w = args.height, args.width
    if args.func == 'yuv':
        if args.yuvlayout in ('yuyv', 'yvyu'):
            img = np.full((h, w * 2), 128, np.uint8)
            img[:, 0::2] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        elif args.yuvlayout == 'nv12':
            img = np.full((h * 3 // 2, w), 128, np.uint8)
            img[:h, :] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        elif args.yuvlayout in ('yv12', 'i420'):
            img = np.full((h * 3 // 2, w), 128, np.uint8)
            img[:h, :] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        elif args.yuvlayout == 'yuv422_row':
            img = np.full((h, w * 2), 128, np.uint8)
        elif args.yuvlayout == 'yuv422_col':
            img = np.full((h * 2, w), 128, np.uint8)
        else:
            img = np.full((h, w * 3), 128, np.uint8)
        img = np.ascontiguousarray(img)
        channels = 1
    elif args.yuv:
        img = np.zeros((h * 3 // 2, w), np.uint8)
        img[:h, :] = 128
        img[h:, :] = 128
        img = np.ascontiguousarray(img)
        channels = 1
    else:
        channels = args.channels or (4 if args.alpha else 3)
        img = np.zeros((h, w, channels), np.uint8)
        img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        img[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        if channels == 4:
            img[:, :, 3] = 255
        if args.fmt == 1:  # BGR(A) request: swap R and B
            img[:, :, [0, 2]] = img[:, :, [2, 0]]
        img = np.ascontiguousarray(img)

    torch_tensor = None
    if args.gpu:
        import torch
        left_t = torch.from_numpy(img).cuda()
        right_t = torch.from_numpy(np.ascontiguousarray(np.flip(img, axis=0))).cuda()
        torch_tensor = (left_t, right_t)
        left, right = left_t.data_ptr(), right_t.data_ptr()
    elif args.same:
        left = right = img.ctypes.data
    elif args.pair:
        right_img = np.ascontiguousarray(np.flip(img, axis=0))
        pair_buf = np.concatenate([img.ravel(), right_img.ravel()])
        left = pair_buf.ctypes.data
        right = left + img.nbytes
        torch_tensor = pair_buf
    else:
        left_img = np.ascontiguousarray(img)
        right_img = np.ascontiguousarray(np.flip(img, axis=0))
        left, right = left_img.ctypes.data, right_img.ctypes.data

    statuses = []
    if args.ts == 'epoch':
        base_ts = time.time_ns()
    elif args.ts == 'mono':
        base_ts = time.monotonic_ns()
    else:
        base_ts = int(args.ts)
    try:
        for i in range(args.frames):
            s = stream_fn(0, C.c_void_p(left), C.c_void_p(right),
                          base_ts + i * (1_000_000_000 // args.fps),
                          0, 0, 0, 1, 0, 0, 0)
            statuses.append(s)
            time.sleep(1.0 / args.fps)
    finally:
        lib.close_streamer(0)
    print('stream', statuses, flush=True)
    print('ok_frames', sum(1 for s in statuses if s == 0), 'dropped',
          sum(1 for s in statuses if s == -1), 'failed',
          sum(1 for s in statuses if s not in (0, -1)), flush=True)
    return 0 if any(s == 0 for s in statuses) else 1


if __name__ == '__main__':
    raise SystemExit(main())

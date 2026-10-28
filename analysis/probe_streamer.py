"""Probe the 5.5 sim runtime's init_streamer with one parameter combination.

Run one combination per process because a bad ABI/encoder path can hard-crash.
The struct matches the official 5.5 types_c.h (100 bytes) plus zero padding, in
case the runtime's build expects additional trailing fields.
"""
import argparse
import ctypes as C
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDK = Path(os.environ.get(
    'FSAI_ZED_SDK_ROOT',
    r'L:\save\userdata\Downloads\ZED_SDK_extract\app'))
DLL = ROOT / 'zed_runtime' / 'sl_zed64.dll'
SERIAL = 110307008


class Params(C.Structure):
    _fields_ = [
        ('mode', C.c_int), ('imu_cam_q', C.c_float * 4), ('imu_cam_t', C.c_float * 3),
        ('image_width', C.c_int), ('image_height', C.c_int), ('codec_type', C.c_int),
        ('port', C.c_ushort), ('fps', C.c_int), ('serial_number', C.c_int),
        ('alpha_channel_included', C.c_bool), ('input_format', C.c_int),
        ('verbose', C.c_bool), ('transport_layer_mode', C.c_int), ('bitrate', C.c_int),
        ('chunk_size', C.c_ushort), ('gop_size', C.c_int), ('gpu_input', C.c_bool),
        ('stream_depth', C.c_bool), ('depth_width', C.c_int), ('depth_height', C.c_int),
        ('depth_bitrate', C.c_int),
        ('_pad', C.c_byte * 156),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--width', type=int, default=960)
    p.add_argument('--height', type=int, default=600)
    p.add_argument('--codec', type=int, default=1, help='0 H264, 1 H265')
    p.add_argument('--transport', type=int, default=0)
    p.add_argument('--port', type=int, default=30100)
    p.add_argument('--fps', type=int, default=15)
    p.add_argument('--serial', type=int, default=SERIAL)
    p.add_argument('--dll', type=Path, default=DLL)
    args = p.parse_args()
    dll = args.dll
    handles = [os.add_dll_directory(str(dll.parent))]
    if (SDK / 'bin').is_dir():
        handles.append(os.add_dll_directory(str(SDK / 'bin')))
    lib = C.CDLL(str(dll))
    lib.getZEDSDKRuntimeVersion_C.argtypes = [C.POINTER(C.c_int)] * 3
    lib.getZEDSDKRuntimeVersion_C.restype = C.c_int
    v = [C.c_int() for _ in range(3)]
    lib.getZEDSDKRuntimeVersion_C(*(C.byref(x) for x in v))
    print('runtime', tuple(x.value for x in v), 'struct', C.sizeof(Params), flush=True)
    lib.is_sn_valid.argtypes = [C.c_int]
    lib.is_sn_valid.restype = C.c_bool
    print('sn_valid', bool(lib.is_sn_valid(args.serial)), flush=True)

    # Supply the same virtual-camera config the bridge writes.
    settings = Path(os.environ.get('PROGRAMDATA', r'C:\ProgramData')) / 'Stereolabs/settings'
    settings.mkdir(parents=True, exist_ok=True)
    conf = settings / f'SN{args.serial}.conf'
    source = next((ROOT / 'outputs/sdk_three_way').glob('bridge_*/settings/*.conf'), None)
    if source and not conf.exists():
        shutil.copyfile(source, conf)
    print('conf', conf, conf.exists(), flush=True)

    params = Params()
    params.mode = 1
    params.imu_cam_q[:] = (0, 0, 0, 1)
    params.image_width, params.image_height = args.width, args.height
    params.codec_type, params.port, params.fps = args.codec, args.port, args.fps
    params.serial_number = args.serial
    params.alpha_channel_included, params.input_format = True, 0
    params.verbose, params.transport_layer_mode = True, args.transport
    params.bitrate, params.chunk_size, params.gop_size = 16000, 4096, -1

    lib.init_streamer.argtypes = [C.c_int, C.POINTER(Params)]
    lib.init_streamer.restype = C.c_int
    lib.close_streamer.argtypes = [C.c_int]
    print('calling init_streamer...', flush=True)
    status = lib.init_streamer(0, C.byref(params))
    print('init_status', status, flush=True)
    if status == 0:
        lib.close_streamer(0)
        print('closed', flush=True)
    return 0 if status == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())

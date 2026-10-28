"""Run the real ZED SDK over the bundled UE stereo pair (simulation stream).

Requires Windows + NVIDIA + ZED SDK 5.5.x.  This launcher only prepares the
commands the kit's experimental bridge expects and records why a step stopped;
all SDK work is done by scripts/run_sdk.py.  It never writes an SDK depth array.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDK_DEFAULT = Path(r'L:\save\userdata\Downloads\ZED_SDK_extract\app')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk-root', type=Path,
                   default=Path(os.environ.get('ZED_SDK_ROOT_DIR', SDK_DEFAULT)))
    p.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'sdk_three_way')
    args = p.parse_args()
    sdk = args.sdk_root
    report = {'sdk_root': str(sdk), 'exists': sdk.is_dir(), 'steps': []}

    def step(name, ok, detail):
        report['steps'].append({'step': name, 'ok': bool(ok), 'detail': detail})
        print(f'[{ "OK" if ok else "STOP" }] {name}: {detail}', flush=True)
        return ok

    if not step('sdk installed', sdk.is_dir(), str(sdk)):
        (args.output).mkdir(parents=True, exist_ok=True)
        (args.output / 'preflight.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8')
        return 1

    pyz = sdk / 'get_python_api.py'
    get_pyz = pyz.is_file()
    step('get_python_api.py', get_pyz, str(pyz))

    def pyzed_version():
        r = subprocess.run(
            [sys.executable, '-c',
             'import pyzed.sl as sl; print(sl.Camera.get_sdk_version())'],
            capture_output=True, text=True,
            env=dict(os.environ, ZED_SDK_ROOT_DIR=str(sdk)))
        return r.returncode == 0, (r.stdout + r.stderr).strip()[-1000:]

    ok, detail = pyzed_version()
    if not ok and get_pyz:
        r = subprocess.run([sys.executable, str(pyz)], capture_output=True,
                           text=True, env=dict(os.environ, ZED_SDK_ROOT_DIR=str(sdk)))
        step('pyzed install', r.returncode == 0,
             (r.stdout + r.stderr).strip()[-2000:])
        ok, detail = pyzed_version()
    step('pyzed import', ok, detail)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'preflight.json').write_text(json.dumps(report, indent=2),
                                                encoding='utf-8')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

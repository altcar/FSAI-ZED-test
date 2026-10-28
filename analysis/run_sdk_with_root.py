"""Launch the kit's experimental SDK bridge against an extracted SDK root.

The SDK installer was extracted (not installed) to keep this machine's global
state untouched.  This wrapper puts the SDK's bin directory on the DLL search
path, points the bridge at it, and then runs scripts/run_sdk.py unchanged.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDK = Path(os.environ.get(
    'FSAI_ZED_SDK_ROOT',
    r'L:\save\userdata\Downloads\ZED_SDK_extract\app'))

os.environ['ZED_SDK_ROOT_DIR'] = str(SDK)
os.add_dll_directory(str(SDK / 'bin'))
sys.path.insert(0, str(ROOT / 'scripts'))

import run_sdk  # noqa: E402

if __name__ == '__main__':
    sys.argv = [str(ROOT / 'scripts' / 'run_sdk.py')] + sys.argv[1:]
    run_sdk.main()

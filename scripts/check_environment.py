import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor'))
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'outputs/yolo_config'))


def main():
    import torch
    import torchvision
    import numpy as np
    import cv2
    import ultralytics
    report = {'python': sys.version, 'os': platform.platform(), 'torch': torch.__version__,
              'torchvision': torchvision.__version__, 'numpy': np.__version__, 'opencv': cv2.__version__,
              'ultralytics': ultralytics.__version__, 'ultralytics_source': ultralytics.__file__,
              'torch_cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available()}
    if torch.cuda.is_available():
        report['gpu'] = [{'name': torch.cuda.get_device_name(i),
                          'vram_gib': torch.cuda.get_device_properties(i).total_memory / 1024**3}
                         for i in range(torch.cuda.device_count())]
        # A real operation catches driver/kernel failures that availability alone misses.
        report['cuda_operation_ok'] = float((torch.ones(8, device='cuda') * 2).sum().cpu()) == 16
    try:
        report['nvidia_smi'] = subprocess.run(['nvidia-smi'], text=True, capture_output=True,
                                             timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        report['nvidia_smi'] = 'unavailable'
    try:
        import pyzed.sl as sl
        report['zed_sdk'] = sl.Camera.get_sdk_version()
    except (ImportError, OSError) as e:
        report['zed_sdk'] = 'not installed: ' + str(e)
    print(json.dumps(report, indent=2))
    (ROOT / 'outputs').mkdir(exist_ok=True)
    (ROOT / 'outputs/environment.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not report['cuda_available']:
        print('CUDA unavailable: CPU comparison works; ZED SDK requires the Windows NVIDIA environment.')


if __name__ == '__main__':
    main()

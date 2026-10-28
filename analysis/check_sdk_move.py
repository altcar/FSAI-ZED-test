import filecmp
import sys
from pathlib import Path

src = Path(r'C:\ZED_SDK_extract')
dst = Path(r'L:\save\userdata\Downloads\ZED_SDK_extract')

missing, mismatch = [], []
src_files = [p for p in src.rglob('*') if p.is_file()]
for p in src_files:
    rel = p.relative_to(src)
    q = dst / rel
    try:
        size = p.stat().st_size
    except FileNotFoundError:
        continue  # source disappeared during the move
    if not q.is_file():
        missing.append(str(rel))
    elif size != q.stat().st_size:
        mismatch.append(str(rel))

print('src files', len(src_files), 'missing at dst', len(missing), 'size mismatch', len(mismatch))
for m in (missing + mismatch)[:10]:
    print(' ', m)
sys.exit(0 if not missing and not mismatch else 1)

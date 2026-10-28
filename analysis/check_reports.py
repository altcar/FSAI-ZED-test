import re
from pathlib import Path

for name in ('ZED_SDK_SIMULATION_RESULT.md', 'STEREO_DEPTH_VALIDATION.md'):
    path = Path(name)
    text = path.read_text(encoding='utf-8')
    blocks, cur = [], []
    for line in text.splitlines():
        if line.strip().startswith('|'):
            cur.append(line)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    bad = []
    for i, block in enumerate(blocks):
        counts = {len(row.strip().strip('|').split('|')) for row in block}
        if len(counts) != 1:
            bad.append((i, counts))
    links = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text)
    missing = [link for link in links if not Path(link).is_file()]
    print(f'{name}: tables={len(blocks)} bad={bad} images={len(links)} missing={missing}')

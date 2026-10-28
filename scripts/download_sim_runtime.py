"""Download the independently versioned sim runtime used by the pinned official repo.

No SDK installation or license acceptance is automated. Compatibility is checked
by sim_streamer.py --probe on Windows, not inferred from the archive filename.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = ('https://stereolabs.sfo2.cdn.digitaloceanspaces.com/utils/zed_isaac_sim/'
       'sl_zed/5.2.0/sl_zed64_windows_x86_64_5.2.0.tar.gz')


def main():
    destination = ROOT / 'zed_runtime'
    if list(destination.glob('*.dll')):
        raise FileExistsError('zed_runtime already contains DLLs; preserve or move them before downloading')
    print('Official runtime download:', URL, flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / 'runtime.tar.gz'
        urllib.request.urlretrieve(URL, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        staging = Path(tmp) / 'unpacked'
        staging.mkdir()
        with tarfile.open(archive, 'r:gz') as tf:
            # Refuse links, device files, Windows drive names and traversal paths.
            for member in tf.getmembers():
                path = Path(member.name.replace('\\', '/'))
                if (path.is_absolute() or '..' in path.parts or ':' in str(path)
                        or not (member.isfile() or member.isdir())):
                    raise ValueError('Unsafe archive member: ' + member.name)
            tf.extractall(staging, filter='data')
        dlls = list(staging.rglob('sl_zed64.dll'))
        if len(dlls) != 1:
            raise RuntimeError('Archive does not contain a unique sl_zed64.dll')
        # Keep all dependent DLLs/data beside the primary library.
        shutil.copytree(dlls[0].parent, destination, dirs_exist_ok=True)
        (destination / 'download_receipt.json').write_text(json.dumps(
            {'url': URL, 'sha256': digest, 'official_source_commit':
             'f9160973f321ec743f626188244cd9702d85bec6',
             'runtime_compatibility': 'unverified until --probe; do not treat archive label as SDK version'},
            indent=2), encoding='utf-8')
    print('Downloaded. Run 04 to probe actual SDK version and C exports. A rejected version needs a matching vendor runtime.')


if __name__ == '__main__':
    main()

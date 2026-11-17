"""Build the EOTDL model snapshot from a clean, tagged checkout plus the large files.

    python scripts/prepare_snapshot.py --tag v1.0.0
Writes dist/<tag>/sentinel-landcover-inference/ with independent copies of every
Git-tracked file, the three verified checkpoints and the test predictions, plus
release-manifest.json. Never uploads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NAME = 'sentinel-landcover-inference'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / 'dist' / args.tag / NAME
    if git('status', '--porcelain', '--untracked-files=no'):
        raise SystemExit('Commit source changes before building a release snapshot.')
    commit = git('rev-parse', 'HEAD')
    if commit != git('rev-parse', f'{args.tag}^{{commit}}'):
        raise SystemExit(f'Check out {args.tag} to build its snapshot.')
    readme = (ROOT / 'README.md').read_text()
    if not readme.startswith('---\n') or not re.search(rf'^name: {NAME}$', readme.split('---', 2)[1], re.M):
        raise SystemExit('README.md must start with EOTDL metadata naming this model.')
    sources = [ROOT / p for p in git('ls-files').splitlines() if p != '.gitignore']
    large = [(ROOT / 'checkpoints' / m['filename'], m['sha256'])
             for m in json.loads((ROOT / 'provenance/checkpoints.json').read_text())['models']]
    large += [(ROOT / p['released'], p['released_sha256'])
              for p in json.loads((ROOT / 'provenance/test_predictions.json').read_text())['predictions'].values()]
    for path, expected in large:
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f'Missing or changed large file: {path}')
        sources.append(path)
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for source in sources:
        if source.is_symlink() or not source.is_file():
            raise SystemExit(f'Expected a regular file: {source}')
        relative = source.relative_to(ROOT)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Copy rather than link, so the snapshot cannot change through the working tree.
        shutil.copyfile(source, target)
        entries.append({'path': relative.as_posix(), 'bytes': target.stat().st_size, 'sha256': digest(target)})
    manifest = {'name': NAME, 'release': args.tag, 'github_commit': commit,
                'github_release': f'https://github.com/alalbiol/{NAME}/releases/tag/{args.tag}',
                'eotdl_model': f'https://www.eotdl.com/models/{NAME}', 'eotdl_version': None,
                'files': entries,
                'note': 'Lists every payload file except itself. The EOTDL version is assigned on upload.'}
    (output / 'release-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'Snapshot ready: {output}; {len(entries)} files; '
          f'{sum(e["bytes"] for e in entries) / 1e6:.0f} MB. Nothing uploaded.')


if __name__ == '__main__':
    main()

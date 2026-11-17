"""Check the downloaded checkpoints against provenance/checkpoints.json."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from landcover.model import sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, default=ROOT / 'checkpoints')
    args = parser.parse_args()
    missing = []
    for entry in json.loads((ROOT / 'provenance/checkpoints.json').read_text())['models']:
        path = args.checkpoint_dir / entry['filename']
        if not path.is_file():
            missing.append(entry['filename'])
            continue
        if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise SystemExit(f'Checkpoint hash mismatch: {path}')
        print(f'{entry["model_id"]}: verified ({entry["bytes"] / 1e6:.0f} MB)')
    if missing:
        raise SystemExit(f'Missing {missing}; download the EOTDL model snapshot (README.md).')


if __name__ == '__main__':
    main()

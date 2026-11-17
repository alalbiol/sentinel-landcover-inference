"""Predict the 19 BigEarthNet land-cover classes for Sentinel-1/2 patches.

Single patch (BigEarthNet layout: one GeoTIFF per band):
    python inference.py --model terrafm_concat --s1-dir <S1 patch folder> --s2-dir <S2 patch folder>
Batch (CSV with s1_path and s2_path columns; optional s1_name/s2_name; paths may be
relative to --root-dir):
    python inference.py --model terrafm_concat --csv patches.csv --root-dir <data root>
Array input (.npz with `imagery` (14, H, W): VV, VH in dB, then B01-B12 L2A DN):
    python inference.py --model terrafm_concat --npz chip.npz

Writes a CSV with one row per patch and one sigmoid probability per class (columns in
alphabetical class order), as the AI4DIS ESA inference script did. Multi-label: each
probability is independent; the reported F1 values use a 0.5 threshold.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from landcover.data import prepare, read_geotiff_patch, read_npz_patch
from landcover.model import CLASSES, MODELS, ROOT, load_model


def patches(args):
    if args.npz:
        for path in args.npz:
            yield {'id': Path(path).stem, 's1': str(path), 's2': str(path)}, read_npz_patch(path)
    elif args.csv:
        root = Path(args.root_dir) if args.root_dir else Path()
        with open(args.csv) as f:
            for row in csv.DictReader(f):
                s1, s2 = root / row['s1_path'], root / row['s2_path']
                yield ({'id': row.get('s1_name') or s1.name, 's1': str(s1), 's2': str(s2)},
                       read_geotiff_patch(s1, s2, row.get('s1_name'), row.get('s2_name')))
    else:
        yield ({'id': Path(args.s1_dir).name, 's1': args.s1_dir, 's2': args.s2_dir},
               read_geotiff_patch(args.s1_dir, args.s2_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=list(MODELS), default='terrafm_concat')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--s1-dir')
    source.add_argument('--csv')
    source.add_argument('--npz', nargs='+')
    parser.add_argument('--s2-dir')
    parser.add_argument('--root-dir')
    parser.add_argument('--output', type=Path, default=Path('outputs/predictions.csv'))
    parser.add_argument('--checkpoint-dir', type=Path, default=ROOT / 'checkpoints')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.s1_dir and not args.s2_dir:
        parser.error('--s1-dir needs --s2-dir')
    if not (args.checkpoint_dir / MODELS[args.model]['file']).is_file():
        raise SystemExit(f'Checkpoint {MODELS[args.model]["file"]} not found in {args.checkpoint_dir}; '
                         'download it from EOTDL (see README.md).')
    model = load_model(args.model, args.checkpoint_dir, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['patch_id', 's1_source', 's2_source', *CLASSES])
        batch, meta = [], []

        def flush():
            with torch.no_grad():
                probabilities = torch.sigmoid(model(torch.stack(batch).to(args.device))).cpu().numpy()
            for info, p in zip(meta, probabilities):
                writer.writerow([info['id'], info['s1'], info['s2'], *np.char.mod('%.8g', p)])
            batch.clear()
            meta.clear()

        for info, channels in patches(args):
            batch.append(prepare(channels))
            meta.append(info)
            count += 1
            if len(batch) == args.batch_size:
                flush()
        if batch:
            flush()
    print(f'{args.model}: {count} patch(es) -> {args.output}')


if __name__ == '__main__':
    main()

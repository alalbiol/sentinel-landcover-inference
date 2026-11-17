"""Test-set metrics from the released BigEarthNet test predictions.

    python scripts/evaluate.py            # all models; needs the EOTDL prediction files
    python scripts/evaluate.py --write    # (authors) refresh evaluation/test_metrics.json

Computes, per model, macro/micro average precision, F1 at probability 0.5 and ROC AUC
over the 19 classes, plus per-class values, on all 189,275 BigEarthNet v2 test patches
(cloud, shadow and snow patches included). Results are compared with
evaluation/test_metrics.json and written to outputs/evaluation/.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from landcover.model import CLASSES, MODELS  # noqa: E402

THRESHOLD = 0.5
TOLERANCE = 1e-9


def read_gz(path):
    with gzip.open(path, 'rt', newline='') as f:
        return list(csv.reader(f))


def labels():
    rows = read_gz(ROOT / 'evaluation/test_labels.csv.gz')[1:]
    index = {name: i for i, name in enumerate(CLASSES)}
    y = np.zeros((len(rows), len(CLASSES)), np.int32)
    for i, (_, _, names) in enumerate(rows):
        for name in filter(None, names.split('|')):
            y[i, index[name]] = 1
    return [r[0] for r in rows], y


def scores(model_id, order):
    rows = read_gz(ROOT / f'evaluation/predictions/{model_id}_test.csv.gz')
    if rows[0][2:] != CLASSES:
        raise SystemExit(f'{model_id}: unexpected class columns')
    by_name = {r[0]: r[2:] for r in rows[1:]}
    return np.array([by_name[name] for name in order], dtype=np.float64)


def metrics(y, p):
    predicted = (p >= THRESHOLD).astype(int)
    summary = {'patches': int(len(y)),
               'ap_macro': average_precision_score(y, p, average='macro'),
               'ap_micro': average_precision_score(y, p, average='micro'),
               'f1_macro': f1_score(y, predicted, average='macro', zero_division=0),
               'f1_micro': f1_score(y, predicted, average='micro', zero_division=0),
               'auc_macro': roc_auc_score(y, p, average='macro'),
               'auc_micro': roc_auc_score(y, p, average='micro')}
    per_class = {name: {'positives': int(y[:, k].sum()),
                        'ap': average_precision_score(y[:, k], p[:, k]),
                        'f1': f1_score(y[:, k], predicted[:, k], zero_division=0),
                        'auc': roc_auc_score(y[:, k], p[:, k])} for k, name in enumerate(CLASSES)}
    return {'summary': summary, 'per_class': per_class}


def close(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    return abs(a - b) <= TOLERANCE


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', default=list(MODELS))
    parser.add_argument('--write', action='store_true', help='overwrite evaluation/test_metrics.json')
    args = parser.parse_args()
    order, y = labels()
    results = {}
    for model_id in args.models:
        results[model_id] = metrics(y, scores(model_id, order))
        s = results[model_id]['summary']
        print(f'{model_id:15s} AP macro {s["ap_macro"]:.4f}  AP micro {s["ap_micro"]:.4f}  '
              f'F1 macro {s["f1_macro"]:.4f}  F1 micro {s["f1_micro"]:.4f}  AUC macro {s["auc_macro"]:.4f}')
    out = ROOT / 'outputs/evaluation'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'test_metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    with (out / 'per_class_metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'class', 'positives', 'ap', 'f1', 'auc'])
        for model_id, result in results.items():
            for name, v in result['per_class'].items():
                writer.writerow([model_id, name, v['positives'], f'{v["ap"]:.6f}', f'{v["f1"]:.6f}',
                                 f'{v["auc"]:.6f}'])
    reference = ROOT / 'evaluation/test_metrics.json'
    if args.write:
        reference.write_text(json.dumps(results, indent=2) + '\n')
        print(f'Wrote {reference}')
        return
    saved = json.loads(reference.read_text())
    differing = [m for m in results if not close(results[m], saved[m])]
    if differing:
        raise SystemExit(f'FAIL: metrics differ from evaluation/test_metrics.json for {differing}')
    print(f'PASS: test metrics reproduced for {", ".join(results)} (outputs/evaluation/).')


if __name__ == '__main__':
    main()

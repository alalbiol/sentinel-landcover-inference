"""Package the original BigEarthNet test-set predictions and labels for release.

    python scripts/prepare_test_predictions.py --source-dir /path/to/AI4DIS/use_case_2

The AI4DIS files hold absolute local paths; this keeps only the patch names (the last
path component, which equals the BigEarthNet patch name) and copies every probability
string unchanged. Writes evaluation/test_labels.csv.gz and
evaluation/predictions/<model>_test.csv.gz, and records source hashes in
provenance/test_predictions.json. Run once by the authors; users start from the outputs.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from landcover.model import CLASSES  # noqa: E402

SOURCES = {'terrafm_early': 'terrafm_earlyfusion_test_predictions.csv',
           'terrafm_concat': 'terrafm_concat_test_predictions.csv',
           'vit_base': 'vit_base_preds.csv'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_gz(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 makes the archive byte-reproducible.
    with gzip.GzipFile(path, 'wb', mtime=0) as raw, \
            io.TextIOWrapper(raw, encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-dir', type=Path, required=True)
    args = parser.parse_args()
    csv.field_size_limit(1 << 24)
    record = {'test_set': {'file': 'test_set.csv', 'sha256': sha256(args.source_dir / 'test_set.csv')},
              'predictions': {}}
    with (args.source_dir / 'test_set.csv').open() as f:
        labels = [(r['s1_name'], r['s2_name'], re.findall(r"'([^']*)'", r['labels']))
                  for r in csv.DictReader(f)]
    unknown = {name for _, _, names in labels for name in names} - set(CLASSES)
    if unknown:
        raise SystemExit(f'Unknown label names: {unknown}')
    names = {s1 for s1, _, _ in labels}
    write_gz(ROOT / 'evaluation/test_labels.csv.gz', ['s1_name', 's2_name', 'labels'],
             [(s1, s2, '|'.join(sorted(n))) for s1, s2, n in labels])
    record['test_set']['patches'] = len(labels)
    record['test_set']['released_sha256'] = sha256(ROOT / 'evaluation/test_labels.csv.gz')
    for model_id, filename in SOURCES.items():
        source = args.source_dir / filename
        with source.open() as f:
            reader = csv.reader(f)
            header = next(reader)
            if header[:2] != ['S1_Path', 'S2_Path'] or header[2:] != CLASSES:
                raise SystemExit(f'{filename}: unexpected columns')
            rows = [[Path(r[0]).name, Path(r[1]).name, *r[2:]] for r in reader]
        if {r[0] for r in rows} != names or len(rows) != len(labels):
            raise SystemExit(f'{filename}: patches differ from the test set')
        target = ROOT / f'evaluation/predictions/{model_id}_test.csv.gz'
        write_gz(target, ['s1_name', 's2_name', *CLASSES], rows)
        record['predictions'][model_id] = {'source': filename, 'source_sha256': sha256(source),
                                           'rows': len(rows), 'released': target.relative_to(ROOT).as_posix(),
                                           'released_bytes': target.stat().st_size,
                                           'released_sha256': sha256(target)}
        print(f'{model_id}: {len(rows):,} rows -> {target.relative_to(ROOT)}')
    (ROOT / 'provenance/test_predictions.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()

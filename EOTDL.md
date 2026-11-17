# EOTDL model snapshot

This is an EOTDL **model entry** named `sentinel-landcover-inference`. The root README
carries its EOTDL metadata. GitHub holds the code, configurations, test labels and
reference metrics. The snapshot is the complete tagged Git tree **plus**:

- `checkpoints/`: three inference-only weight files (1.39 GB);
- `evaluation/predictions/`: per-patch test predictions for each model (56 MB).

## GitHub and EOTDL version mapping

GitHub release `v1.0.0` identifies the code. The EOTDL version number is assigned on
upload and is not derived from the Git tag. After upload, record it in
`provenance/releases/v1.0.0.json` and the README.

## Build the snapshot

From a clean checkout of the tag, with the large files present:

```bash
git checkout v1.0.0
python scripts/verify_checkpoints.py
python scripts/evaluate.py
python scripts/prepare_snapshot.py --tag v1.0.0
```

The snapshot is written to `dist/v1.0.0/sentinel-landcover-inference/` as independent
copies, with `release-manifest.json` (commit, tag, per-file SHA-256). Existing folders are
never overwritten. The large files can be rebuilt from the original AI4DIS files:

```bash
python scripts/export_checkpoints.py --source-dir /path/to/ESA_MODELS
python scripts/prepare_test_predictions.py --source-dir /path/to/AI4DIS/use_case_2
```

## Upload

Use a separate Python 3.12+ environment for the EOTDL client:

```bash
python3.12 -m venv /tmp/eotdl-uploader
source /tmp/eotdl-uploader/bin/activate
python -m pip install eotdl
eotdl auth login
eotdl models ingest -p ./dist/v1.0.0/sentinel-landcover-inference
```

Push the GitHub repository and release first, so the `source` link resolves. After
upload, check the published hashes against `release-manifest.json`, add the release
record, run `python scripts/update_github_about.py` (after `gh auth login`), and update
the AI4DIS project README's EOTDL resources table.

## Download

```bash
eotdl auth login
eotdl models get sentinel-landcover-inference --version <N> --path ./downloads
```

Run `python scripts/verify_checkpoints.py` before inference.

[Official model ingestion instructions](https://www.eotdl.com/docs/models/ingest)

---
name: sentinel-landcover-inference
authors:
  - Alberto Albiol
  - Miquel Obrador-Reina
  - Juan B. Pedro
  - Albert Fernández
license: MIT AND CC-BY-4.0
source: https://github.com/alalbiol/sentinel-landcover-inference/tree/v1.0.0
---

# Sentinel land-cover inference

Classify Sentinel-1/2 image patches into the 19 BigEarthNet land-cover classes (multi-label) with three AI4DIS models built on the TerraFM-B vision transformer. They differ in how they combine Sentinel-1 radar and Sentinel-2 optical bands:

| Model | Fusion | Initialisation | Test AP (macro / micro) |
|---|---|---|---:|
| `terrafm_concat` (default) | Late: S1 and S2 encoded separately, features concatenated | TerraFM-B weights | **0.718 / 0.868** |
| `terrafm_early` | Early: all 14 bands in one patch embedding | TerraFM-B weights | 0.673 / 0.846 |
| `vit_base` | Late: S1 and S2 features averaged | Random (trained from scratch) | 0.655 / 0.829 |

Test metrics cover all 189,275 patches of the BigEarthNet v2 test split, including those flagged for cloud, shadow or snow. See [MODEL_CARD.md](MODEL_CARD.md) for per-class results, how they were obtained, and limitations.

**Release:** [GitHub v1.0.0](https://github.com/alalbiol/sentinel-landcover-inference/releases/tag/v1.0.0). **EOTDL model:** [`sentinel-landcover-inference`](https://www.eotdl.com/models/sentinel-landcover-inference); its version is recorded in `provenance/releases/` after upload.

| | GitHub | EOTDL snapshot |
|---|---|---|
| Inference, evaluation and training code; configurations; test labels and metrics | ✓ | ✓ |
| Three inference-only checkpoints (1.39 GB) | | ✓ |
| Per-patch test predictions for each model (189,275 × 19, 56 MB) | | ✓ |

The early-fusion backbone also underlies the AI4DIS forest-to-agriculture screening model ([forest-to-agriculture-inference](https://github.com/alalbiol/forest-to-agriculture-inference)).

## Quick start

Use Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# CPU-only: python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Download the EOTDL snapshot (below) or copy its `checkpoints/` folder into this repository, then:

```bash
python scripts/verify_checkpoints.py
python inference.py --model terrafm_concat --s1-dir <S1 patch folder> --s2-dir <S2 patch folder> \
  --output outputs/predictions.csv
```

The output CSV has one row per patch and one independent probability per class, with columns in alphabetical class order. The reported F1 scores use a threshold of 0.5.

## Input

- **BigEarthNet layout (GeoTIFF):** an S1 folder with `*_VV.tif` and `*_VH.tif`, and an S2 folder with `*_B01.tif` … `*_B12.tif` (including `B8A`). Use `--s1-dir/--s2-dir` for one patch, or `--csv` with columns `s1_path, s2_path` (and optionally `s1_name, s2_name`; paths may be relative to `--root-dir`) for many.
- **Array (`--npz`):** `imagery` (14, H, W) in the order VV, VH, B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12.

Units: Sentinel-1 backscatter in dB (linear values are converted automatically), and Sentinel-2 L2A surface reflectance as DN (about reflectance × 10,000). Do not rescale. Coarser bands are bilinearly upsampled to the finest band's grid, the stack is resized to 224 × 224, and each band is standardised with the BigEarthNet statistics. This is exactly the preprocessing that produced the test predictions. BigEarthNet patches are 1.2 × 1.2 km.

## Evaluation

```bash
python -m pip install -r requirements-eval.txt
python scripts/evaluate.py
```

Recomputes macro/micro AP, F1 at 0.5, ROC AUC and per-class metrics from the released test predictions (EOTDL), and checks them against `evaluation/test_metrics.json`.

## Training

[training/README.md](training/README.md) contains the training script and the configuration of each released model. It needs BigEarthNet v2 through ConfigILM and, for the TerraFM models, the public TerraFM-B weights. The code is provided as used and was not re-run for this release.

## Repository guide

```text
inference.py      Predict the 19 classes for GeoTIFF patches, a CSV of patches, or .npz arrays
landcover/        Model architectures (TerraFM-B, three fusions), input reading and preprocessing
training/         Training script and the three model configurations
evaluation/       Test labels (BigEarthNet v2 test split) and reference metrics; predictions on EOTDL
checkpoints/      (EOTDL) Inference-only weights
scripts/          Checkpoint export and verification, test evaluation, snapshot preparation
provenance/       Checkpoint hashes and training metadata, equivalence and metric checks
tests/            Input, model, training and label tests (python -m unittest discover -s tests)
```

## EOTDL download

```bash
eotdl auth login
eotdl models get sentinel-landcover-inference --path ./downloads
```

The snapshot is self-contained: it holds the full tagged repository plus `checkpoints/` and `evaluation/predictions/`. See [EOTDL.md](EOTDL.md).

## Licence and attribution

- **Code and model weights:** [MIT](LICENSE). The weights are derived from TerraFM-B (MIT), except `vit_base`, which was trained from scratch.
- **Released test predictions and metrics:** [CC BY 4.0](LICENSE-DATA).
- **Third-party components:** `landcover/terrafm.py` is the TerraFM architecture (MIT), with parts adapted from DINO (Apache-2.0). The test labels come from BigEarthNet (CDLA-Permissive-1.0). See [THIRD_PARTY.md](THIRD_PARTY.md).

Prepared within AI4DIS, funded by the European Space Agency under Contract No. 4000147555/25/I-DT. The Universitat Politècnica de València participated as a subcontractor to EarthPulse S.L., the prime contractor. The views expressed are those of the authors and do not necessarily reflect the official position of the European Space Agency.

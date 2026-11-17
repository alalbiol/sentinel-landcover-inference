# Sentinel land-cover models: model card, v1.0.0

## Task

Multi-label classification of Sentinel-1/2 patches into the 19 BigEarthNet land-cover classes, which are derived from CORINE Land Cover. Each class gets an independent sigmoid probability. Input is one co-registered Sentinel-1 GRD (VV, VH) and Sentinel-2 L2A (12 bands) acquisition, in the style of BigEarthNet patches (1.2 × 1.2 km).

## Models

All three share the TerraFM-B architecture (ViT-B/16, 12 blocks, 768-d, 224 × 224 input) and a classification head of LayerNorm, Dropout 0.1 and Linear. They differ in how they fuse S1 and S2, and in how they were initialised:

| Model | Fusion | Initialisation | Training | Saved epoch | Validation AP macro / micro |
|---|---|---|---|---:|---:|
| `terrafm_concat` | S1 (radar embedding) and S2 (L2A embedding) encoded separately by the shared backbone; class tokens concatenated (1,536-d) | TerraFM-B | 10 epochs, backbone frozen for the first 5 | 7th | 0.800 / 0.876 |
| `terrafm_early` | All 14 bands through one new early-fusion patch embedding | TerraFM-B | 10 epochs, backbone frozen for the first 5 except the new embedding | 9th | 0.776 / 0.867 |
| `vit_base` | As concat, but the class tokens are averaged (768-d) | Random | From scratch; 100-epoch schedule, label smoothing 0.1, rotation augmentation | 14th | 0.760 / 0.856 |

Common settings: BigEarthNet v2 via ConfigILM, all 14 bands, AdamW with cosine schedule and warm-up, BCE loss, horizontal/vertical flips, mixed precision, batch 32. The checkpoint with the best validation macro AP was kept. Full configurations are in `training/configs/`. `vit_base` used the project's from-scratch configuration.

The released files contain the model tensors only, copied unchanged from the original training checkpoints, which also held the optimiser state (`provenance/checkpoints.json`).

## Test results

These are the BigEarthNet v2 test split as used in the project: 189,275 patches, including those flagged for cloud, shadow or snow. Predictions were produced with the project's GeoTIFF inference path, which `inference.py` reproduces exactly. F1 uses a threshold of 0.5.

| Model | AP macro | AP micro | F1 macro | F1 micro | AUC macro | AUC micro |
|---|---:|---:|---:|---:|---:|---:|
| `terrafm_concat` | **0.718** | **0.868** | **0.638** | **0.764** | **0.948** | **0.968** |
| `terrafm_early` | 0.673 | 0.846 | 0.600 | 0.749 | 0.931 | 0.961 |
| `vit_base` | 0.655 | 0.829 | 0.583 | 0.730 | 0.909 | 0.955 |

Per-class average precision:

| Class | Test positives | `terrafm_concat` | `terrafm_early` | `vit_base` |
|---|---:|---:|---:|---:|
| Agro-forestry areas | 9,967 | 0.864 | 0.792 | 0.795 |
| Arable land | 62,669 | 0.909 | 0.887 | 0.873 |
| Beaches, dunes, sands | 187 | 0.255 | 0.195 | 0.110 |
| Broad-leaved forest | 45,928 | 0.854 | 0.838 | 0.840 |
| Coastal wetlands | 340 | 0.231 | 0.157 | 0.137 |
| Complex cultivation patterns | 25,649 | 0.665 | 0.653 | 0.641 |
| Coniferous forest | 85,639 | 0.960 | 0.955 | 0.947 |
| Industrial or commercial units | 2,996 | 0.504 | 0.371 | 0.337 |
| Inland waters | 42,883 | 0.931 | 0.892 | 0.869 |
| Inland wetlands | 11,748 | 0.621 | 0.557 | 0.502 |
| Land principally occupied by agriculture, with significant areas of natural vegetation | 46,173 | 0.719 | 0.659 | 0.660 |
| Marine waters | 20,323 | 0.963 | 0.948 | 0.921 |
| Mixed forest | 84,456 | 0.920 | 0.909 | 0.903 |
| Moors, heathland and sclerophyllous vegetation | 4,502 | 0.623 | 0.586 | 0.592 |
| Natural grassland and sparsely vegetated areas | 4,020 | 0.547 | 0.480 | 0.481 |
| Pastures | 31,247 | 0.860 | 0.855 | 0.842 |
| Permanent crops | 5,855 | 0.627 | 0.598 | 0.543 |
| Transitional woodland, shrub | 64,884 | 0.765 | 0.721 | 0.721 |
| Urban fabric | 16,727 | 0.827 | 0.740 | 0.729 |

Per-class F1 and AUC are in `evaluation/test_metrics.json`.

**Validation and test numbers are not directly comparable.** Validation used ConfigILM's data pipeline during training. Test used separate GeoTIFF reading with bilinear resampling, and includes cloud, shadow and snow patches, which the training data loader is configured to identify. Macro AP falls from 0.78–0.80 (validation) to 0.66–0.72 (test). The cause of the gap has not been investigated; possible contributors are the patch selection and the preprocessing differences.

## Verification of this release

- **Code equivalence:** `inference.py` and `landcover/` reproduce the original AI4DIS inference script bit for bit. Preprocessed tensors are identical on a BigEarthNet-layout patch, and the maximum probability difference is 0.0 for all three models (`provenance/equivalence.json`).
- **Metrics:** `scripts/evaluate.py` reproduces the project's reported test metrics from the released predictions, within 3 × 10⁻¹⁶.
- **Training code:** for each configuration, one CPU training step runs, and its checkpoint loads strictly into the inference architecture (`tests/`). Full training was not re-run.
- **Not yet done:** re-running inference on bundled BigEarthNet sample patches against the stored predictions. Sample patches will be added in a later version.

## Limitations

- Trained and tested on BigEarthNet: ten European countries, 2017–2018, with labels from CORINE Land Cover 2018. Performance elsewhere, in other years, or on composites rather than single acquisitions is unknown.
- Labels describe CORINE classes over a 1.2 km patch, not pixel-level land cover. Rare classes (beaches, coastal wetlands, industrial units) have low AP.
- Probabilities are not calibrated. The 0.5 threshold is a convention, not an optimised operating point.
- There is no cloud masking. Cloudy optical inputs degrade predictions.
- Intended for research and screening; not validated for regulatory decisions.

# Training

`train.py` is the AI4DIS script that trained the released models, trimmed to the TerraFM-B
architecture. The schedule, losses, freezing, checkpoint selection and checkpoint format
are unchanged. It is **provided as used and was not re-run for this release**. A re-run
will not be bit-identical, because of GPU nondeterminism, mixed precision and data order.

| Model | Configuration | Starts from |
|---|---|---|
| `terrafm_concat` | `configs/terrafm_concat.yaml` | TerraFM-B weights; late fusion by concatenation |
| `terrafm_early` | `configs/terrafm_early_fusion.yaml` | TerraFM-B weights plus a new early-fusion patch embedding |
| `vit_base` | `configs/vit_base_scratch.yaml` | Random initialisation; late fusion by averaging (the project's from-scratch configuration) |

## Setup

GPU recommended:

```bash
python -m pip install -r requirements.txt -r training/requirements.txt
```

1. **BigEarthNet v2** in ConfigILM's layout: an LMDB of encoded patches, `metadata.parquet`, and the parquet of patches with snow, cloud or shadow. See https://bigearth.net and https://github.com/lhackel-tub/ConfigILM.
2. **TerraFM-B weights** (for the two fine-tuned models): https://github.com/mbzuai-oryx/TerraFM.
3. Set the paths in the chosen configuration (`/path/to/...`), then run:

```bash
python training/train.py --config training/configs/terrafm_concat.yaml
```

## Schedule

- **Bands:** ConfigILM band set `ALL` (VV, VH, B01–B12), the order `inference.py` uses.
- **Fine-tuned models** (`terrafm_concat`, `terrafm_early`): AdamW (lr 1e-4, weight decay 0.05), cosine schedule with 5 warm-up epochs, 10 epochs.
  - For the first 5 epochs the backbone is frozen. Only the head trains, plus the new patch embedding in the early-fusion model.
  - Then everything is unfrozen, and the optimiser and schedule restart.
- **`vit_base`** (from scratch): lr 3e-4, 10 warm-up epochs, a 100-epoch schedule, label smoothing 0.1, and ±15° rotation augmentation.
- **All models:** BCE loss, flips, mixed precision, gradient clip 1.0, batch 32, seed 42.
- **Saving:** the checkpoint with the best validation macro AP is saved as `<save_dir>/best_model.pth`.

## Use a trained checkpoint

`best_model.pth` holds the model and optimiser state. Its `model_state_dict` loads
strictly into `landcover.model.LandCoverClassifier(<fusion>)`. To use it with
`inference.py`, save it in the release format:

```python
import torch
saved = torch.load('best_model.pth', map_location='cpu', weights_only=False)
torch.save({'state_dict': saved['model_state_dict'], 'fusion': 'concat', 'classes': 19},
           'checkpoints/terrafm_late_fusion_concat.pt')
```

Use `fusion` `early`, `concat` or `mean`, and the matching file name in `landcover/model.py`.

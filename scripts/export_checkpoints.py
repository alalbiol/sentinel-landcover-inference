"""Export inference-only weights from the original AI4DIS fine-tuning checkpoints.

    python scripts/export_checkpoints.py --source-dir /path/to/ESA_MODELS
The originals also hold the optimiser state, which is dropped. Each source file is
hash-checked, its model tensors are copied unchanged, and the exported file is loaded
strictly into the release architecture and compared tensor for tensor. Sizes, hashes and
training metadata are written to provenance/checkpoints.json.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from landcover.model import EARLY_FUSION_PARAMETERS, MODELS, load_model, sha256  # noqa: E402

SOURCES = {
    'terrafm_early': ('TerraFM_EarlyFusion.pth', '5a7c585e66752364360d54f21c6c4197b87cc85eec670c11650012d86c8d7bdc'),
    'terrafm_concat': ('TerraFM_LateFusion_Concat.pth', '7e666c86f41fb84eb30e6d8d4d22f37e5d48a0dc3772d5430952b6ad93d7a833'),
    'vit_base': ('ViT_base.pth', '909fdb54dd52418e48ac4b24195104403395b5e4b86ffb0154fde0999ac706cb'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'checkpoints')
    args = parser.parse_args()
    record_path = ROOT / 'provenance/checkpoints.json'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for model_id, (filename, expected) in SOURCES.items():
        source = args.source_dir / filename
        digest = sha256(source)
        if digest != expected:
            raise SystemExit(f'{source}: hash differs from the recorded source checkpoint')
        target = args.output_dir / MODELS[model_id]['file']
        if target.exists():
            raise SystemExit(f'{target} exists; remove it to re-export.')
        original = torch.load(source, map_location='cpu', weights_only=False)
        state = {k: v.clone() for k, v in original['model_state_dict'].items()}
        if MODELS[model_id]['fusion'] != 'early':
            # Present only if the source was saved from a later architecture version; unused.
            for name in EARLY_FUSION_PARAMETERS:
                state.pop(f'backbone.{name}', None)
        torch.save({'state_dict': state, 'fusion': MODELS[model_id]['fusion'], 'classes': 19,
                    'source_sha256': digest, 'source_epoch_index': int(original['epoch'])}, target)
        model = load_model(model_id, args.output_dir)
        reloaded = model.state_dict()
        assert reloaded.keys() == state.keys()
        assert all(torch.equal(reloaded[k], state[k]) for k in state)
        records.append({'model_id': model_id, 'filename': target.name, 'bytes': target.stat().st_size,
                        'sha256': sha256(target), 'parameters': int(sum(v.numel() for v in state.values())),
                        'source': {'filename': filename, 'bytes': source.stat().st_size, 'sha256': digest,
                                   'epoch_index': int(original['epoch']),
                                   'validation_metrics': original.get('metrics'),
                                   'dropped_top_level_keys': [k for k in original if k != 'model_state_dict']}})
        print(f'{model_id}: {target.name} ({target.stat().st_size / 1e6:.0f} MB)')
    record_path.write_text(json.dumps({'models': records}, indent=2) + '\n')


if __name__ == '__main__':
    main()

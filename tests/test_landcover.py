"""Input, model, training and label-file tests (CPU). python -m unittest discover -s tests

Tests that need the checkpoints are skipped when they are not downloaded.
"""
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training'))
from landcover.data import BANDS, prepare, read_geotiff_patch, read_npz_patch  # noqa: E402
from landcover.model import CLASSES, MODELS, LandCoverClassifier, load_model  # noqa: E402
import train as training  # noqa: E402

CHECKPOINTS = all((ROOT / 'checkpoints' / m['file']).is_file() for m in MODELS.values())
NATIVE = {'B01': 20, 'B09': 20, 'B05': 60, 'B06': 60, 'B07': 60, 'B8A': 60, 'B11': 60, 'B12': 60}


def bigearthnet_like_patch(folder: Path):
    """Write a synthetic patch with BigEarthNet's file layout and band resolutions."""
    import rasterio
    from rasterio.transform import from_origin
    rng = np.random.default_rng(0)
    s1, s2 = folder / 'S1_patch', folder / 'S2_patch'
    for directory in (s1, s2):
        directory.mkdir()
    for band in BANDS:
        size = NATIVE.get(band, 120)
        values = rng.normal(-15, 4, (size, size)) if band in ('VV', 'VH') else rng.uniform(0, 4000, (size, size))
        directory = s1 if band in ('VV', 'VH') else s2
        with rasterio.open(directory / f'{directory.name}_{band}.tif', 'w', driver='GTiff', height=size,
                           width=size, count=1, dtype='float32', crs='EPSG:32633',
                           transform=from_origin(0, 0, 1200 / size, 1200 / size)) as dst:
            dst.write(values.astype(np.float32), 1)
    return s1, s2


class InputTest(unittest.TestCase):
    def test_geotiff_patch_with_mixed_resolutions(self):
        with tempfile.TemporaryDirectory() as tmp:
            channels = read_geotiff_patch(*bigearthnet_like_patch(Path(tmp)))
        self.assertEqual([c.shape[0] for c in channels], [NATIVE.get(b, 120) for b in BANDS])
        tensor = prepare(channels)
        self.assertEqual(tuple(tensor.shape), (14, 224, 224))
        self.assertTrue(torch.isfinite(tensor).all())

    def test_npz_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chip.npz'
            np.savez(path, imagery=np.ones((14, 44, 44), np.float32))
            channels = read_npz_patch(path)
        self.assertAlmostEqual(float(channels[0][0, 0]), 10 * np.log10(1 + 1e-6), places=6)  # linear S1 -> dB

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            prepare([np.zeros((10, 10))] * 13)
        bad = [np.zeros((10, 10)) for _ in BANDS]
        bad[4][0, 0] = np.nan
        with self.assertRaises(ValueError):
            prepare(bad)


class ModelTest(unittest.TestCase):
    def test_architectures(self):
        widths = {'early': 768, 'concat': 1536, 'mean': 768}
        for fusion, width in widths.items():
            model = LandCoverClassifier(fusion).eval()
            self.assertEqual(model.classifier[0].normalized_shape, (width,))
            with torch.no_grad():
                self.assertEqual(tuple(model(torch.randn(1, 14, 224, 224)).shape), (1, len(CLASSES)))

    @unittest.skipUnless(CHECKPOINTS, 'checkpoints not downloaded')
    def test_checkpoints_load_strictly(self):
        records = {m['model_id']: m for m in json.loads((ROOT / 'provenance/checkpoints.json').read_text())['models']}
        for model_id in MODELS:
            model = load_model(model_id)
            self.assertEqual(sum(p.numel() for p in model.parameters()), records[model_id]['parameters'])


class TrainingTest(unittest.TestCase):
    def test_one_step_per_configuration(self):
        for name in ('terrafm_early_fusion', 'terrafm_concat', 'vit_base_scratch'):
            config = training.Config(ROOT / f'training/configs/{name}.yaml')
            config.PRETRAINED_MODEL_PATH = None
            config.MIXED_PRECISION = False
            config.NUM_EPOCHS = 1
            with tempfile.TemporaryDirectory() as tmp:
                config.SAVE_DIR = Path(tmp)
                torch.manual_seed(0)
                model = training.create_model(config)
                loader = [(torch.randn(2, 14, 224, 224), torch.tensor([[1, 0] * 9 + [1], [0, 1] * 9 + [0]]))]
                optimizer = training.create_optimizer(model, config)
                scheduler = training.create_scheduler(optimizer, config, len(loader))
                loss, _ = training.train_epoch(model, loader, training.create_criterion(config), optimizer,
                                               scheduler, None, 'cpu', config, 0, lambda values: None)
                self.assertTrue(np.isfinite(loss), name)
                metrics = training.validate(model, loader, training.create_criterion(config), 'cpu', 0, config)
                training.save_checkpoint(model, optimizer, 0, metrics, config, 'best_model.pth')
                saved = torch.load(Path(tmp) / 'best_model.pth', weights_only=False)['model_state_dict']
                LandCoverClassifier(config.FUSION_TYPE).load_state_dict(saved, strict=True)


class LabelFileTest(unittest.TestCase):
    def test_test_labels(self):
        with gzip.open(ROOT / 'evaluation/test_labels.csv.gz', 'rt') as f:
            rows = [line.rstrip('\n').split(',', 2) for line in f][1:]
        self.assertEqual(len(rows), 189275)
        names = {n for row in rows for n in row[2].strip('"').split('|') if n}
        self.assertEqual(names, set(CLASSES))


if __name__ == '__main__':
    unittest.main()

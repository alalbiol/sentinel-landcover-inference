"""Reading Sentinel-1/2 patches and preparing the 14-channel model input.

Identical to the AI4DIS `ESA_inference_script.ESAInferenceDataset` that produced the
reported test predictions:
  1. read VV, VH and B01-B12 (one GeoTIFF per band, BigEarthNet layout);
  2. Sentinel-1 linear backscatter is converted to dB (bands already in dB are kept);
  3. coarser bands are bilinearly upsampled to the finest band's grid;
  4. the 14-band stack is bilinearly resized to 224 x 224;
  5. each band is standardised with the BigEarthNet statistics below.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

S1_BANDS = ['VV', 'VH']
S2_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
BANDS = S1_BANDS + S2_BANDS
INPUT_SIZE = 224
# BigEarthNet statistics in BANDS order (S1 in dB, S2 L2A surface-reflectance DN).
MEAN = np.array([-12.64386368, -19.35255814, 361.07678223, 438.37207031, 614.05566406,
                 588.40960693, 942.84332275, 1769.93164062, 2049.55151367, 2193.29199219,
                 2235.55664062, 2241.45532227, 1568.22680664, 997.73248291], np.float32)
STD = np.array([5.1334939, 5.5905056, 575.06872559, 607.02685547, 603.29681396,
                684.56884766, 738.43267822, 1100.45605469, 1275.80541992, 1369.3717041,
                1356.54406738, 1316.39331055, 1070.16125488, 813.52764893], np.float32)


def _band_path(folder: Path, prefix: str | None, band: str) -> Path:
    if prefix is not None and (folder / f'{prefix}_{band}.tif').exists():
        return folder / f'{prefix}_{band}.tif'
    matches = list(folder.glob(f'*{band}.tif'))
    if len(matches) != 1:
        raise FileNotFoundError(f'Band {band} not found unambiguously in {folder}')
    return matches[0]


def read_geotiff_patch(s1_folder, s2_folder, s1_prefix=None, s2_prefix=None) -> list[np.ndarray]:
    """Return the 14 bands (S1 in dB) at their native shapes."""
    import rasterio
    s1_folder, s2_folder = Path(s1_folder), Path(s2_folder)
    channels = []
    for band in S1_BANDS:
        with rasterio.open(_band_path(s1_folder, s1_prefix, band)) as src:
            image = src.read(1)
        if np.min(image) >= 0:
            image = 10 * np.log10(image + 1e-6)
        channels.append(image)
    for band in S2_BANDS:
        with rasterio.open(_band_path(s2_folder, s2_prefix, band)) as src:
            channels.append(src.read(1))
    return channels


def read_npz_patch(path) -> list[np.ndarray]:
    """An .npz with `imagery` (14, H, W) in BANDS order (S1 in dB, S2 L2A DN)."""
    with np.load(path) as saved:
        imagery = saved['imagery']
    if imagery.ndim != 3 or imagery.shape[0] != len(BANDS):
        raise ValueError(f'{path}: expected imagery (14, H, W), got {imagery.shape}')
    channels = list(imagery)
    for index in range(len(S1_BANDS)):
        if np.min(channels[index]) >= 0:
            channels[index] = 10 * np.log10(channels[index] + 1e-6)
    return channels


def prepare(channels: list[np.ndarray]) -> torch.Tensor:
    """14 bands at native resolutions -> normalised (14, 224, 224) float tensor."""
    if len(channels) != len(BANDS):
        raise ValueError(f'expected {len(BANDS)} bands, got {len(channels)}')
    if not all(np.isfinite(c).all() for c in channels):
        raise ValueError('input contains non-finite values')
    height = max(c.shape[0] for c in channels)
    width = max(c.shape[1] for c in channels)
    resized = []
    for image in channels:
        if image.shape != (height, width):
            tensor = torch.from_numpy(np.asarray(image)).float()[None, None]
            image = F.interpolate(tensor, size=(height, width), mode='bilinear',
                                  align_corners=False).squeeze().numpy()
        resized.append(image)
    data = np.stack(resized, axis=0)
    if data.shape[1:] != (INPUT_SIZE, INPUT_SIZE):
        data = F.interpolate(torch.from_numpy(data).float()[None], size=(INPUT_SIZE, INPUT_SIZE),
                             mode='bilinear', align_corners=False)[0].numpy()
    data = (data - MEAN.reshape(-1, 1, 1)) / STD.reshape(-1, 1, 1)
    return torch.from_numpy(data).float()

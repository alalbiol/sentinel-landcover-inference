# Sources and third-party components

- **TerraFM** (`landcover/terrafm.py`): architecture code from https://github.com/mbzuai-oryx/TerraFM, MIT licence, © 2025 TerraFM Authors, Mohamed bin Zayed University of Artificial Intelligence. It is copied with one AI4DIS addition, the 14-channel early-fusion patch embedding. The TerraFM licence also covers the TerraFM-B weights from which `terrafm_early` and `terrafm_concat` were fine-tuned. Cite Danish et al., *TerraFM: A Scalable Foundation Model for Unified Multisensor Earth Observation*, arXiv:2506.06281.
- **DINO**: parts of `landcover/terrafm.py` are adapted from https://github.com/facebookresearch/dino, © Facebook, Inc. and its affiliates, Apache License 2.0. Those portions remain under Apache-2.0.
- **BigEarthNet v2** (Clasen et al., reBEN; https://bigearth.net), Community Data License Agreement – Permissive 1.0. It provides the training data and the test-split labels redistributed in `evaluation/test_labels.csv.gz`. The labels derive from CORINE Land Cover 2018 (Copernicus Land Monitoring Service). The imagery is modified Copernicus Sentinel data and is not redistributed here.
- **ConfigILM** (MIT, © 2023 Leonard Wayne Hackel; https://github.com/lhackel-tub/ConfigILM): BigEarthNet v2 loader and band statistics used by `training/train.py`; an installed dependency, not bundled.
- **Weights & Biases**: optional logging in `training/train.py`, disabled by default.
- **PyTorch, NumPy, rasterio, scikit-learn**: installed dependencies under their own licences.

No training dataset, manuscript, credentials or local data paths are included in this repository or the snapshot.

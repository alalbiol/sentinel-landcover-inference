"""TerraFM-B land-cover classifiers for the 19 BigEarthNet classes.

Three Sentinel-1/2 fusion variants share one ViT-B/16 backbone architecture (TerraFM-B):

  early   VV, VH and B01-B12 enter one 14-channel patch embedding together.
  concat  S1 and S2 are encoded separately by the same backbone (S1 and S2 L2A patch
          embeddings); the two 768-d class tokens are concatenated (1536-d).
  mean    as concat, but the two class tokens are averaged (768-d).

A LayerNorm -> Dropout -> Linear head gives 19 logits; sigmoid gives per-class
probabilities (multi-label). This reproduces the forward pass of the AI4DIS
`EarthObservationClassifier` for these configurations, with the same parameter names.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn

from .terrafm import terrafm_base

ROOT = Path(__file__).resolve().parents[1]
EMBED_DIM = 768
FUSIONS = ('early', 'concat', 'mean')
EARLY_FUSION_PARAMETERS = ('patch_embed.conv2d_s1_s2.weight', 'patch_embed.conv2d_s1_s2.bias',
                           'patch_embed.s1_s2_embed')

# BigEarthNet-19 classes, alphabetical (BigEarthNet v2 / ConfigILM order). Output column order.
CLASSES = sorted([
    'Urban fabric', 'Industrial or commercial units', 'Arable land', 'Permanent crops', 'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas', 'Broad-leaved forest', 'Coniferous forest', 'Mixed forest',
    'Natural grassland and sparsely vegetated areas', 'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland, shrub', 'Beaches, dunes, sands', 'Inland wetlands', 'Coastal wetlands',
    'Inland waters', 'Marine waters'])

MODELS = {
    'terrafm_early': {'fusion': 'early', 'file': 'terrafm_early_fusion.pt',
                      'description': 'TerraFM-B, early S1/S2 fusion, fine-tuned from TerraFM-B weights'},
    'terrafm_concat': {'fusion': 'concat', 'file': 'terrafm_late_fusion_concat.pt',
                       'description': 'TerraFM-B, late fusion (concatenated S1 and S2 class tokens), '
                                      'fine-tuned from TerraFM-B weights'},
    'vit_base': {'fusion': 'mean', 'file': 'vit_base_scratch.pt',
                 'description': 'TerraFM-B architecture trained from random initialisation, '
                                'late fusion (averaged S1 and S2 class tokens)'},
}


class LandCoverClassifier(nn.Module):
    def __init__(self, fusion: str, num_classes: int = 19, dropout: float = 0.1,
                 pretrained_path=None, freeze_backbone: bool = False):
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f'fusion must be one of {FUSIONS}')
        self.fusion = fusion
        self.backbone = terrafm_base()
        if fusion != 'early':
            # The late-fusion models were trained without the early-fusion embedding;
            # removing it lets their weights load strictly.
            del self.backbone.patch_embed.conv2d_s1_s2
            del self.backbone.patch_embed.s1_s2_embed
        if pretrained_path:
            load_terrafm_weights(self.backbone, pretrained_path)
        width = 2 * EMBED_DIM if fusion == 'concat' else EMBED_DIM
        self.classifier = nn.Sequential(nn.LayerNorm(width), nn.Dropout(dropout),
                                        nn.Linear(width, num_classes))
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze the backbone; for early fusion the new fusion embedding stays trainable."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if self.fusion == 'early':
            for parameter in self.backbone.patch_embed.conv2d_s1_s2.parameters():
                parameter.requires_grad = True
            self.backbone.patch_embed.s1_s2_embed.requires_grad = True

    def unfreeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def features(self, x: torch.Tensor, is_l2a: bool = True) -> torch.Tensor:
        if x.shape[1] != 14:
            raise ValueError(f'expected 14 channels (VV, VH, B01-B12), got {x.shape[1]}')
        if self.fusion == 'early':
            return self.backbone.forward(x, is_l2a=is_l2a)
        s1 = self.backbone.forward(x[:, :2], is_l2a=False)
        s2 = self.backbone.forward(x[:, 2:], is_l2a=is_l2a)
        return torch.cat((s1, s2), dim=1) if self.fusion == 'concat' else (s1 + s2) / 2.0

    def forward(self, x: torch.Tensor, is_l2a: bool = True) -> torch.Tensor:
        return self.classifier(self.features(x, is_l2a))

    def get_parameter_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'frozen': total - trainable}


def load_terrafm_weights(backbone: nn.Module, path) -> None:
    """Load public TerraFM-B weights; layers absent from them stay newly initialised."""
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    state = {k: v for k, v in state.items() if not k.startswith('head')}
    report = backbone.load_state_dict(state, strict=False)
    print(f'Loaded TerraFM weights from {path}: {report}')


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_model(model_id: str, checkpoint_dir=ROOT / 'checkpoints', device='cpu') -> LandCoverClassifier:
    spec = MODELS[model_id]
    saved = torch.load(Path(checkpoint_dir) / spec['file'], map_location='cpu', weights_only=True)
    if saved['fusion'] != spec['fusion']:
        raise ValueError(f'{spec["file"]} holds a {saved["fusion"]} model, expected {spec["fusion"]}')
    model = LandCoverClassifier(spec['fusion'])
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(device).eval()

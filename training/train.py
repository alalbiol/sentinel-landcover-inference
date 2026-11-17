"""Train the TerraFM-B land-cover classifiers on BigEarthNet v2 (19-class multi-label).

    python training/train.py --config training/configs/terrafm_concat.yaml

This is the AI4DIS training script that produced the released models, trimmed to the
TerraFM-architecture configurations (early, concat and mean S1/S2 fusion). Schedule,
losses, freezing, checkpoint selection and checkpoint format are unchanged; Weights &
Biases logging is optional (`wandb.enabled`). Provided as used; it was not re-run for
this release. A re-run will not be bit-identical (GPU nondeterminism, mixed precision,
data-loader order).

Requires BigEarthNet v2 in ConfigILM's LMDB/parquet layout and, for fine-tuning, the
public TerraFM-B weights; see training/README.md. The checkpoint with the best
validation macro AP is saved as <save_dir>/best_model.pth.
"""
import argparse
from pathlib import Path
import random
import sys

import numpy as np
from sklearn.metrics import average_precision_score, f1_score
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from landcover.model import LandCoverClassifier  # noqa: E402


class Config:
    """Configuration loaded from YAML."""

    def __init__(self, config_path):
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        self._config_dict = config_dict
        self.experiment_name = config_dict.get('experiment_name', 'experiment')
        model_cfg = config_dict.get('model', {})
        self.NUM_CLASSES = model_cfg.get('num_classes', 19)
        self.FUSION_TYPE = model_cfg.get('fusion_type', 'mean')
        pretrained_path = model_cfg.get('pretrained_path')
        self.PRETRAINED_MODEL_PATH = Path(pretrained_path) if pretrained_path else None
        self.FREEZE_BACKBONE = model_cfg.get('freeze_backbone', False)
        self.FREEZE_EPOCHS = model_cfg.get('freeze_epochs', 0)
        self.DROPOUT = model_cfg.get('dropout', 0.1)
        data_cfg = config_dict.get('data', {})
        self.DATA_DIRS = data_cfg.get('data_dirs', {})
        self.BAND_COMBINATION = data_cfg.get('band_combination', 'ALL')
        self.IMG_HEIGHT, self.IMG_WIDTH = data_cfg.get('img_size', [224, 224])
        self.BATCH_SIZE = data_cfg.get('batch_size', 32)
        self.NUM_WORKERS = data_cfg.get('num_workers', 4)
        train_cfg = config_dict.get('training', {})
        self.NUM_EPOCHS = train_cfg.get('num_epochs', 10)
        self.LEARNING_RATE = train_cfg.get('learning_rate', 1e-4)
        self.WEIGHT_DECAY = train_cfg.get('weight_decay', 0.05)
        self.OPTIMIZER = train_cfg.get('optimizer', 'adamw')
        self.SCHEDULER = train_cfg.get('scheduler', 'cosine')
        self.WARMUP_EPOCHS = train_cfg.get('warmup_epochs', 5)
        self.GRADIENT_CLIP = train_cfg.get('gradient_clip')
        self.MIXED_PRECISION = train_cfg.get('mixed_precision', True)
        self.LABEL_SMOOTHING = train_cfg.get('label_smoothing', 0.0)
        aug_cfg = config_dict.get('augmentation', {})
        self.HORIZONTAL_FLIP = aug_cfg.get('horizontal_flip', 0.5)
        self.VERTICAL_FLIP = aug_cfg.get('vertical_flip', 0.5)
        self.ROTATION_DEGREES = aug_cfg.get('rotation_degrees', 0)
        ckpt_cfg = config_dict.get('checkpoint', {})
        self.SAVE_DIR = Path(ckpt_cfg.get('save_dir', 'checkpoints'))
        self.SAVE_FREQ = ckpt_cfg.get('save_freq', 5)
        wandb_cfg = config_dict.get('wandb', {})
        self.USE_WANDB = wandb_cfg.get('enabled', False)
        self.WANDB_PROJECT = wandb_cfg.get('project', 'terrafm-bigearthnet')
        self.WANDB_ENTITY = wandb_cfg.get('entity')
        self.WANDB_RUN_NAME = wandb_cfg.get('run_name')
        self.SEED = config_dict.get('seed', 42)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(config):
    pretrained = config.PRETRAINED_MODEL_PATH
    if pretrained is not None and not pretrained.exists():
        raise FileNotFoundError(f'TerraFM-B weights not found: {pretrained}')
    model = LandCoverClassifier(config.FUSION_TYPE, config.NUM_CLASSES, config.DROPOUT, pretrained,
                                config.FREEZE_BACKBONE)
    params = model.get_parameter_counts()
    print(f"\nModel ({config.FUSION_TYPE} fusion) parameters: total {params['total']:,}, trainable {params['trainable']:,}, "
          f"frozen {params['frozen']:,}")
    return model


def create_optimizer(model, config):
    if config.OPTIMIZER == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    if config.OPTIMIZER == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9,
                               weight_decay=config.WEIGHT_DECAY)
    raise ValueError(f'Unknown optimizer: {config.OPTIMIZER}')


def create_scheduler(optimizer, config, steps_per_epoch):
    if config.SCHEDULER == 'cosine':
        total_steps = config.NUM_EPOCHS * steps_per_epoch
        warmup_steps = config.WARMUP_EPOCHS * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if config.SCHEDULER == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return None


def create_criterion(config):
    if config.LABEL_SMOOTHING > 0:
        class BCEWithLabelSmoothing(nn.Module):
            def __init__(self, smoothing):
                super().__init__()
                self.smoothing = smoothing
                self.bce = nn.BCEWithLogitsLoss()

            def forward(self, pred, target):
                return self.bce(pred, target * (1 - self.smoothing) + 0.5 * self.smoothing)
        return BCEWithLabelSmoothing(config.LABEL_SMOOTHING)
    return nn.BCEWithLogitsLoss()


def unpack(batch, device):
    if isinstance(batch, dict):
        return batch['image'].to(device), batch['label'].to(device)
    return batch[0].to(device), batch[1].to(device)


def train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, config, epoch, log):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{config.NUM_EPOCHS} [train]')
    for batch_idx, batch in enumerate(pbar):
        images, labels = unpack(batch, device)
        optimizer.zero_grad()
        if config.MIXED_PRECISION:
            with torch.amp.autocast('cuda'):
                outputs = model(images, is_l2a=True)
                loss = criterion(outputs, labels.float())
            scaler.scale(loss).backward()
            if config.GRADIENT_CLIP is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, is_l2a=True)
            loss = criterion(outputs, labels.float())
            loss.backward()
            if config.GRADIENT_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
        if scheduler is not None and config.SCHEDULER == 'cosine':
            scheduler.step()
        total_loss += loss.item()
        correct += ((torch.sigmoid(outputs) > 0.5).float() == labels).sum().item()
        total += labels.numel()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})
        log({'train/loss': loss.item(), 'train/lr': optimizer.param_groups[0]['lr'],
             'train/step': epoch * len(loader) + batch_idx})
    return total_loss / len(loader), 100 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, config):
    model.eval()
    total_loss, labels_all, outputs_all = 0.0, [], []
    for batch in tqdm(loader, desc=f'Epoch {epoch + 1}/{config.NUM_EPOCHS} [val]'):
        images, labels = unpack(batch, device)
        outputs = model(images, is_l2a=True)
        total_loss += criterion(outputs, labels.float()).item()
        labels_all.append(labels.cpu().numpy())
        outputs_all.append(torch.sigmoid(outputs).cpu().numpy())
    labels_all, outputs_all = np.concatenate(labels_all), np.concatenate(outputs_all)
    predictions = (outputs_all > 0.5).astype(int)
    return {'loss': total_loss / len(loader),
            'ap_macro': average_precision_score(labels_all, outputs_all, average='macro'),
            'ap_micro': average_precision_score(labels_all, outputs_all, average='micro'),
            'f1_macro': f1_score(labels_all, predictions, average='macro', zero_division=0),
            'f1_micro': f1_score(labels_all, predictions, average='micro', zero_division=0)}


def save_checkpoint(model, optimizer, epoch, metrics, config, filename):
    config.SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'metrics': metrics},
               config.SAVE_DIR / filename)
    print(f'Checkpoint saved to {config.SAVE_DIR / filename}')


def data_loaders(config):
    from configilm.extra.BENv2_utils import band_combi_to_mean_std, resolve_band_combi
    from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet
    from torch.utils.data import DataLoader
    import torchvision.transforms as T

    bands = resolve_band_combi(config.BAND_COMBINATION)  # 'ALL': VV, VH, B01-B12 (B8A before B09)
    print(f'Bands: {bands}')
    mean, std = band_combi_to_mean_std(bands)
    augment = [T.RandomHorizontalFlip(p=config.HORIZONTAL_FLIP), T.RandomVerticalFlip(p=config.VERTICAL_FLIP)]
    if config.ROTATION_DEGREES > 0:
        augment.append(T.RandomRotation(degrees=config.ROTATION_DEGREES))
    normalise = T.Normalize(mean=mean.tolist(), std=std.tolist())
    size = (len(bands), config.IMG_HEIGHT, config.IMG_WIDTH)
    train = BENv2DataSet(data_dirs=config.DATA_DIRS, split='train',
                         transform=T.Compose(augment + [normalise]), img_size=size)
    val = BENv2DataSet(data_dirs=config.DATA_DIRS, split='validation', transform=normalise, img_size=size)
    print(f'Train samples: {len(train):,}; validation samples: {len(val):,}')
    kwargs = dict(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, pin_memory=True)
    return DataLoader(train, shuffle=True, **kwargs), DataLoader(val, shuffle=False, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = Config(args.config)
    set_seed(config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Experiment {config.experiment_name} on {device}')
    log = lambda values: None  # noqa: E731
    if config.USE_WANDB:
        import wandb
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY,
                   name=config.WANDB_RUN_NAME or config.experiment_name, config=config._config_dict)
        log = wandb.log
    train_loader, val_loader = data_loaders(config)
    model = create_model(config).to(device)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    criterion = create_criterion(config)
    scaler = torch.amp.GradScaler('cuda') if config.MIXED_PRECISION else None
    best_val_ap = 0
    for epoch in range(config.NUM_EPOCHS):
        if epoch == config.FREEZE_EPOCHS and config.FREEZE_BACKBONE:
            # As in the original run: unfreeze everything and restart optimiser and schedule.
            model.unfreeze_backbone()
            optimizer = create_optimizer(model, config)
            scheduler = create_scheduler(optimizer, config, len(train_loader))
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler,
                                            scaler, device, config, epoch, log)
        metrics = validate(model, val_loader, criterion, device, epoch, config)
        if scheduler is not None and config.SCHEDULER == 'step':
            scheduler.step()
        log({'epoch': epoch, 'train/epoch_loss': train_loss, 'train/epoch_acc': train_acc,
             **{f'val/{k}': v for k, v in metrics.items()}})
        print(f"Epoch {epoch + 1}: train loss {train_loss:.4f}, val loss {metrics['loss']:.4f}, "
              f"AP macro {metrics['ap_macro']:.4f}, AP micro {metrics['ap_micro']:.4f}")
        record = {'train_loss': train_loss, **metrics}
        if (epoch + 1) % config.SAVE_FREQ == 0:
            save_checkpoint(model, optimizer, epoch, record, config, f'checkpoint_epoch_{epoch + 1}.pth')
        if metrics['ap_macro'] > best_val_ap:
            best_val_ap = metrics['ap_macro']
            save_checkpoint(model, optimizer, epoch, record, config, 'best_model.pth')
    print(f'Best validation AP (macro): {best_val_ap:.4f}')
    if config.USE_WANDB:
        wandb.finish()


if __name__ == '__main__':
    main()

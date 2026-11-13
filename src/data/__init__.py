"""GTSRB loading, preprocessing, and the train / validation / test splits.

Every model (deep models and HOG + SVM) uses the same splits:
  - test:             the official GTSRB test set (12,630 images)
  - train/validation: torchvision's GTSRB training set (26,640 images), split ~80/20

GTSRB training images come in "tracks": ~30 consecutive frames of the same
physical sign. Frames in a track are near-duplicates, so the split is made per
track (stratified by class) - a track goes entirely to train or entirely to
validation. A plain random image split would put near-identical frames in both
sets and make validation scores unrealistically high.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB

from src.config import DATA_DIR, IMG_SIZE, SEED, VAL_FRACTION


def build_transform(img_size=IMG_SIZE):
    """Resize to img_size x img_size and convert to a float tensor in [0, 1]."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def load_gtsrb(img_size=IMG_SIZE, download=True, transform=None):
    """Return the official GTSRB (train, test) datasets."""
    transform = transform or build_transform(img_size)

    root = Path(DATA_DIR)
    root.mkdir(parents=True, exist_ok=True)

    train_full = GTSRB(root=str(root), split="train", download=download, transform=transform)
    test_dataset = GTSRB(root=str(root), split="test", download=download, transform=transform)

    return train_full, test_dataset


def split_train_val(train_full, val_fraction=VAL_FRACTION, seed=SEED):
    """Track-level, class-stratified split of the official training set.

    Training files are named <track>_<frame>.ppm inside one folder per class.
    For each class, a `val_fraction` share of its tracks (at least one) is
    held out for validation. Returns sorted (train_indices, val_indices).
    """
    tracks = defaultdict(lambda: defaultdict(list))     # class -> track -> sample indices
    for idx, (path, label) in enumerate(train_full._samples):
        track = Path(path).stem.split("_")[0]
        tracks[int(label)][track].append(idx)

    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []

    for label in sorted(tracks):
        track_ids = sorted(tracks[label])
        order = rng.permutation(len(track_ids))
        n_val = max(1, round(len(track_ids) * val_fraction))

        for rank, i in enumerate(order):
            target = val_idx if rank < n_val else train_idx
            target.extend(tracks[label][track_ids[i]])

    return sorted(train_idx), sorted(val_idx)


def get_dataloaders(
    batch_size=128,
    num_workers=2,
    img_size=IMG_SIZE,
    val_fraction=VAL_FRACTION,
    seed=SEED,
    download=True
):
    """Load GTSRB and create training, validation and test loaders.

    Batches are (images, labels): images float32 in [0, 1] with shape
    (B, 3, img_size, img_size), labels int64 in 0..42.
    """
    train_full, test_dataset = load_gtsrb(img_size=img_size, download=download)
    train_idx, val_idx = split_train_val(train_full, val_fraction=val_fraction, seed=seed)

    train_dataset = Subset(train_full, train_idx)
    val_dataset = Subset(train_full, val_idx)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        # Keep worker processes alive between epochs (re-spawning them is slow on Windows).
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader

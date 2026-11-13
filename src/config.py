"""Shared paths and training configuration.

Every deep model in the ablation study is trained from the same TrainConfig,
so differences in results come from the architecture, not the training setup.
Change defaults here, not inside individual scripts.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

NUM_CLASSES = 43
IMG_SIZE = 48           # every model sees 48x48 RGB images in [0, 1]
SEED = 42
VAL_FRACTION = 0.20     # fraction of the official training set held out for validation


@dataclass
class TrainConfig:
    seed: int = SEED
    num_classes: int = NUM_CLASSES
    in_channels: int = 3
    img_size: int = IMG_SIZE
    val_fraction: float = VAL_FRACTION

    batch_size: int = 128
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"        # adamw | adam | sgd
    scheduler: str = "cosine"       # cosine | step | none
    label_smoothing: float = 0.0
    grad_clip: Optional[float] = 1.0
    early_stop_patience: int = 7    # epochs without validation macro-F1 improvement
    num_workers: int = 2
    amp: bool = True                # mixed precision; only used on CUDA

    def to_dict(self):
        return asdict(self)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import GTSRB
from torchvision import transforms
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR


def get_dataloaders(
    batch_size=128,
    num_workers=2,
    img_size=48,
    val_fraction=0.20,
    seed=42,
    download=True
):
    """Load GTSRB and create training, validation and test loaders."""

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    root = Path(DATA_DIR)
    root.mkdir(parents=True, exist_ok=True)

    train_full = GTSRB(
        root=str(root),
        split="train",
        download=download,
        transform=transform
    )

    test_dataset = GTSRB(
        root=str(root),
        split="test",
        download=download,
        transform=transform
    )

    targets = [int(label) for _, label in train_full._samples]

    indices = list(range(len(train_full)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=targets
    )

    train_dataset = Subset(train_full, train_idx)
    val_dataset = Subset(train_full, val_idx)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


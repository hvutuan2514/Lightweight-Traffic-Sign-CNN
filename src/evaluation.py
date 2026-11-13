"""Shared evaluation for every deep model."""

import torch
import torch.nn as nn

from src.config import NUM_CLASSES
from src.metrics import compute_metrics


@torch.no_grad()
def predict(model, loader, device):
    """Return (predictions, targets, mean cross-entropy loss) over a loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    preds, targets = [], []
    total_loss, n = 0.0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x).float()

        total_loss += criterion(logits, y).item()
        n += y.size(0)
        preds.append(logits.argmax(1).cpu())
        targets.append(y.cpu())

    return torch.cat(preds).numpy(), torch.cat(targets).numpy(), total_loss / max(n, 1)


def evaluate(model, loader, device, num_classes=NUM_CLASSES):
    """Metrics from src.metrics.compute_metrics plus the mean loss."""
    y_pred, y_true, loss = predict(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, num_classes)
    metrics["loss"] = float(loss)
    return metrics

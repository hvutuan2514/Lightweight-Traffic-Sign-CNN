"""Shared training loop used by every deep model.

Model selection and early stopping use validation macro-F1, the project's
primary metric. The best weights are saved to checkpoints/<run_name>.pt and the
per-epoch history to results/<run_name>_history.json.
"""

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.config import CHECKPOINT_DIR, RESULTS_DIR, TrainConfig, get_device
from src.evaluation import evaluate
from src.metrics import scalar_metrics
from src.models import build_model


def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: TrainConfig):
    params = [p for p in model.parameters() if p.requires_grad]
    name = cfg.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, nesterov=True,
                               weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def build_scheduler(optimizer, cfg: TrainConfig):
    name = cfg.scheduler.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(cfg.epochs // 3, 1), gamma=0.1)
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip=None):
    """One pass over the training set. Returns (mean loss, accuracy)."""
    model.train()
    use_amp = scaler is not None
    total_loss, correct, n = 0.0, 0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)

    return total_loss / n, correct / n


def fit(model, train_loader, val_loader, cfg: TrainConfig, run_name, model_name,
        device=None, checkpoint_dir=CHECKPOINT_DIR, results_dir=RESULTS_DIR):
    """Train with early stopping on validation macro-F1.

    The best weights are restored into `model` before returning, saved to
    <checkpoint_dir>/<run_name>.pt, and the history is written to
    <results_dir>/<run_name>_history.json.
    """
    device = device or get_device()
    model.to(device)
    checkpoint_dir, results_dir = Path(checkpoint_dir), Path(results_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc",
                               "val_macro_f1", "lr", "epoch_time")}
    best_f1, best_epoch, best_state, bad_epochs = -1.0, 0, None, 0
    ckpt_path = checkpoint_dir / f"{run_name}.pt"
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                                device, scaler, cfg.grad_clip)
        val = evaluate(model, val_loader, device, cfg.num_classes)
        if scheduler is not None:
            scheduler.step()
        epoch_time = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["accuracy"])
        history["val_macro_f1"].append(val["macro_f1"])
        history["lr"].append(lr_now)
        history["epoch_time"].append(epoch_time)

        print(f"[{run_name}] epoch {epoch:02d}/{cfg.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val['loss']:.4f} acc {val['accuracy']:.4f} "
              f"macro-F1 {val['macro_f1']:.4f} | lr {lr_now:.2e} | {epoch_time:.1f}s",
              flush=True)

        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = val["macro_f1"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model_name": model_name, "model_state": best_state,
                        "cfg": cfg.to_dict(), "epoch": epoch,
                        "val": scalar_metrics(val)}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                print(f"[{run_name}] early stopping at epoch {epoch} (best epoch {best_epoch})",
                      flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history.update(run=run_name, model=model_name, cfg=cfg.to_dict(),
                   best_epoch=best_epoch, best_val_macro_f1=best_f1,
                   epochs_run=len(history["train_loss"]),
                   total_train_time=time.time() - start)
    (results_dir / f"{run_name}_history.json").write_text(json.dumps(history, indent=2))
    return history


def load_checkpoint(run_name, device=None, checkpoint_dir=CHECKPOINT_DIR):
    """Rebuild a trained model from checkpoints/<run_name>.pt.

    Returns (model in eval mode, checkpoint dict, TrainConfig used for training).
    """
    device = device or get_device()
    ckpt = torch.load(Path(checkpoint_dir) / f"{run_name}.pt", map_location=device)
    cfg = TrainConfig(**ckpt["cfg"])
    model = build_model(ckpt["model_name"], num_classes=cfg.num_classes,
                        img_size=cfg.img_size, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt, cfg

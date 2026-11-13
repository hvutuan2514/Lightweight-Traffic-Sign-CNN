"""Figures saved by the training scripts (PNG, for the report).

Uses matplotlib's Figure API directly (not pyplot), so saving figures never
changes the plotting backend of a notebook that imports this module.
"""

from pathlib import Path

import torch
from matplotlib.figure import Figure

# Colours: train = series 1 (blue), validation = series 2 (orange).
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _style_axis(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Epoch", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def plot_history(history, path, title=None):
    """Loss, accuracy and validation macro-F1 per epoch; the best epoch is marked."""
    epochs = range(1, len(history["train_loss"]) + 1)
    best = history.get("best_epoch")

    fig = Figure(figsize=(13, 3.6), facecolor=SURFACE)
    axes = fig.subplots(1, 3)
    panels = [
        ("Loss", "Cross-entropy", "train_loss", "val_loss"),
        ("Accuracy", "Accuracy", "train_acc", "val_acc"),
    ]
    for ax, (name, ylabel, train_key, val_key) in zip(axes, panels):
        ax.plot(epochs, history[train_key], color=TRAIN_COLOR, linewidth=2, label="Train")
        ax.plot(epochs, history[val_key], color=VAL_COLOR, linewidth=2, label="Validation")
        _style_axis(ax, name, ylabel)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    ax = axes[2]
    ax.plot(epochs, history["val_macro_f1"], color=VAL_COLOR, linewidth=2)
    _style_axis(ax, "Validation macro-F1", "Macro-F1")

    if best:
        best_f1 = history["val_macro_f1"][best - 1]
        ax.scatter([best], [best_f1], s=40, color=VAL_COLOR, edgecolor=SURFACE,
                   linewidth=2, zorder=3)
        ax.annotate(f"best {best_f1:.4f} (epoch {best})", (best, best_f1),
                    xytext=(0, -16), textcoords="offset points", ha="center",
                    fontsize=8, color=INK_SECONDARY)
        for a in axes:
            a.axvline(best, color=AXIS, linewidth=1, linestyle="--", zorder=0)

    if title:
        fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, path)


@torch.no_grad()
def save_stn_examples(model, dataset, device, path, n=10):
    """Top row: input images. Bottom row: the same images after the model's STN.

    Images are taken evenly spaced through `dataset`, so they cover many classes.
    """
    model.eval()
    n = min(n, len(dataset))
    indices = torch.linspace(0, len(dataset) - 1, n).long().tolist()
    x = torch.stack([dataset[i][0] for i in indices]).to(device)
    x_t = model.stn(x)

    fig = Figure(figsize=(1.3 * n, 3.0), facecolor=SURFACE)
    axes = fig.subplots(2, n, squeeze=False)
    for row, images in enumerate((x, x_t)):
        for i in range(n):
            axes[row, i].imshow(images[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
            axes[row, i].axis("off")
    axes[0, 0].set_title("Input", fontsize=9, color=INK_SECONDARY, loc="left")
    axes[1, 0].set_title("After STN", fontsize=9, color=INK_SECONDARY, loc="left")
    fig.tight_layout()
    _save(fig, path)


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())

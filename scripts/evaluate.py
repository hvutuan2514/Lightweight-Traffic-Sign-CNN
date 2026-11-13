"""Evaluate a trained checkpoint on the validation and/or test split.

The data split is rebuilt from the config stored in the checkpoint, so it is
identical to the one the model was trained with.

Run from the repository root:
    python -m scripts.evaluate --run cnn_stn --split val
    python -m scripts.evaluate --run cnn_stn --split test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR, get_device
from src.data import get_dataloaders
from src.evaluation import evaluate
from src.metrics import format_metrics, save_metrics
from src.models import count_parameters
from src.training import load_checkpoint


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="checkpoint name, i.e. checkpoints/<run>.pt")
    p.add_argument("--split", choices=("val", "test", "both"), default="val")
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args(argv)

    device = get_device()
    model, ckpt, cfg = load_checkpoint(args.run, device)

    _, val_loader, test_loader = get_dataloaders(
        batch_size=cfg.batch_size, num_workers=args.num_workers, img_size=cfg.img_size,
        val_fraction=cfg.val_fraction, seed=cfg.seed)
    loaders = {"val": val_loader, "test": test_loader}
    splits = ("val", "test") if args.split == "both" else (args.split,)

    for split in splits:
        metrics = evaluate(model, loaders[split], device, cfg.num_classes)
        save_metrics(metrics, RESULTS_DIR / f"{args.run}_{split}_metrics.json",
                     run=args.run, model=ckpt["model_name"], split=split,
                     parameters=count_parameters(model), best_epoch=ckpt["epoch"], cfg=cfg.to_dict())
        print(f"[{args.run}] {split}: {format_metrics(metrics)}")


if __name__ == "__main__":
    main()

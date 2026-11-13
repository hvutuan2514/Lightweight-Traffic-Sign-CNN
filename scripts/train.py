"""Train one deep model with the shared configuration (src/config.py).

Run from the repository root:
    python -m scripts.train --model basic_cnn
    python -m scripts.train --model cnn_stn
    python -m scripts.train --model cnn_attention
    python -m scripts.train --model cnn_stn_attention
    python -m scripts.train --model resnet18

Outputs
    checkpoints/<run>.pt              best weights (by validation macro-F1)
    results/<run>_history.json        per-epoch loss / accuracy / macro-F1
    results/<run>_val_metrics.json    full validation metrics of the best weights
    results/<run>_test_metrics.json   only with --eval-test
    figures/<run>_history.png         training curves
    figures/<run>_stn.png             inputs vs. STN outputs (STN models only)

The official test set is used only with --eval-test, so it is not touched
while models are still being developed.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FIGURES_DIR, RESULTS_DIR, TrainConfig, get_device
from src.data import get_dataloaders
from src.evaluation import evaluate
from src.metrics import format_metrics, save_metrics
from src.models import MODEL_NAMES, build_model, count_parameters
from src.training import fit, set_seed
from src.visualization import plot_history, save_stn_examples


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    p.add_argument("--run-name", help="defaults to the model name")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--patience", type=int, help="early-stopping patience (epochs)")
    p.add_argument("--num-workers", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--eval-test", action="store_true",
                   help="also evaluate on the official test set (final results only)")

    resnet = p.add_argument_group("ResNet-18 fine-tuning")
    resnet.add_argument("--head-epochs", type=int, default=3,
                        help="phase 1: epochs training only the new head (default 3)")
    resnet.add_argument("--finetune-lr", type=float, default=1e-4,
                        help="phase 2: learning rate for the whole network (default 1e-4)")
    return p.parse_args(argv)


def make_config(args):
    overrides = {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "early_stop_patience": args.patience, "num_workers": args.num_workers, "seed": args.seed,
    }
    return replace(TrainConfig(), **{k: v for k, v in overrides.items() if v is not None})


def main(argv=None):
    args = parse_args(argv)
    run_name = args.run_name or args.model
    cfg = make_config(args)

    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device} | run: {run_name}")
    print(cfg)

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=cfg.batch_size, num_workers=cfg.num_workers, img_size=cfg.img_size,
        val_fraction=cfg.val_fraction, seed=cfg.seed)
    print(f"Images: train {len(train_loader.dataset)} | val {len(val_loader.dataset)} "
          f"| test {len(test_loader.dataset)}")

    model = build_model(args.model, num_classes=cfg.num_classes, img_size=cfg.img_size)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")

    if args.model == "resnet18":
        # Phase 1: train only the new head so the random head does not wreck pretrained features.
        model.freeze_backbone()
        head_cfg = replace(cfg, epochs=args.head_epochs, scheduler="none",
                           early_stop_patience=args.head_epochs)
        fit(model, train_loader, val_loader, head_cfg, run_name=f"{run_name}_head",
            model_name=args.model, device=device)
        # Phase 2: fine-tune everything with a smaller learning rate.
        model.unfreeze_all()
        cfg = replace(cfg, lr=args.finetune_lr)

    history = fit(model, train_loader, val_loader, cfg, run_name=run_name,
                  model_name=args.model, device=device)
    plot_history(history, FIGURES_DIR / f"{run_name}_history.png", title=run_name)

    info = dict(run=run_name, model=args.model, parameters=n_params,
                best_epoch=history["best_epoch"], epochs_run=history["epochs_run"],
                train_time_sec=history["total_train_time"], cfg=cfg.to_dict())

    val_metrics = evaluate(model, val_loader, device, cfg.num_classes)
    save_metrics(val_metrics, RESULTS_DIR / f"{run_name}_val_metrics.json", split="val", **info)
    print(f"Validation (best epoch {history['best_epoch']}): {format_metrics(val_metrics)}")

    if args.eval_test:
        test_metrics = evaluate(model, test_loader, device, cfg.num_classes)
        save_metrics(test_metrics, RESULTS_DIR / f"{run_name}_test_metrics.json", split="test", **info)
        print(f"Test: {format_metrics(test_metrics)}")

    if hasattr(model, "stn"):
        save_stn_examples(model, val_loader.dataset, device, FIGURES_DIR / f"{run_name}_stn.png")


if __name__ == "__main__":
    main()

"""Fast checks that the models and the training pipeline work (no GTSRB needed).

Run from the repository root:
    python -m scripts.smoke_test

The data-split checks also run if GTSRB has already been downloaded to data/.
"""

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import DATA_DIR, IMG_SIZE, NUM_CLASSES, TrainConfig
from src.evaluation import predict
from src.metrics import compute_metrics
from src.models import MODEL_NAMES, STN, build_model, count_parameters
from src.training import fit, load_checkpoint, set_seed


def check_models():
    x = torch.rand(4, 3, IMG_SIZE, IMG_SIZE)
    for name in MODEL_NAMES:
        model = build_model(name, pretrained=False)
        out = model(x)
        assert out.shape == (4, NUM_CLASSES), (name, out.shape)
        nn.functional.cross_entropy(out, torch.zeros(4, dtype=torch.long)).backward()
        print(f"  {name:<18} output {tuple(out.shape)}  parameters {count_parameters(model):>10,}")


def check_stn():
    stn = STN(in_channels=3, img_size=IMG_SIZE)
    x = torch.rand(4, 3, IMG_SIZE, IMG_SIZE)
    y, theta = stn(x, return_theta=True)

    identity = torch.tensor([[1., 0., 0.], [0., 1., 0.]]).expand(4, 2, 3)
    assert torch.allclose(theta, identity), "theta should start as the identity transform"
    assert (y - x).abs().max() < 1e-5, "identity STN should return its input"

    y.square().mean().backward()
    assert stn.fc_loc[2].bias.grad is not None and stn.fc_loc[2].bias.grad.abs().sum() > 0
    print(f"  STN: identity at init, gradients reach the localization net "
          f"({count_parameters(stn):,} parameters)")


def check_resnet_freezing():
    model = build_model("resnet18", pretrained=False)
    model.freeze_backbone()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("backbone.fc") for n in trainable), trainable
    model.train()
    assert not model.backbone.bn1.training, "frozen backbone BatchNorm should stay in eval mode"
    model.unfreeze_all()
    model.train()
    assert model.backbone.bn1.training and all(p.requires_grad for p in model.parameters())
    print("  ResNet-18: head-only phase freezes the backbone (and its BatchNorm stats)")


def check_metrics():
    y = np.array([0, 1, 2, 2])
    m = compute_metrics(y, y, num_classes=NUM_CLASSES)
    assert m["accuracy"] == 1.0 and len(m["confusion_matrix"]) == NUM_CLASSES
    m = compute_metrics(y, np.array([0, 1, 2, 1]), num_classes=3)
    assert m["accuracy"] == 0.75 and m["confusion_matrix"][2] == [0, 1, 1]
    print("  metrics: accuracy / macro scores / confusion matrix")


def check_training_pipeline():
    """Two epochs on random data: fit, checkpoint, reload. The numbers are meaningless."""
    set_seed(0)
    cfg = replace(TrainConfig(), epochs=2, batch_size=32, num_workers=0)
    g = torch.Generator().manual_seed(0)

    def loader(n, shuffle):
        x = torch.rand(n, 3, IMG_SIZE, IMG_SIZE, generator=g)
        y = torch.randint(0, NUM_CLASSES, (n,), generator=g)
        return DataLoader(TensorDataset(x, y), batch_size=cfg.batch_size, shuffle=shuffle)

    train_loader, val_loader = loader(96, True), loader(64, False)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmp:
        model = build_model("cnn_stn_attention")
        history = fit(model, train_loader, val_loader, cfg, run_name="smoke", model_name="cnn_stn_attention",
                      device=device, checkpoint_dir=tmp, results_dir=tmp)
        assert history["epochs_run"] == 2 and (Path(tmp) / "smoke_history.json").exists()

        reloaded, ckpt, reloaded_cfg = load_checkpoint("smoke", device, checkpoint_dir=tmp)
        assert reloaded_cfg == cfg and ckpt["epoch"] == history["best_epoch"]
        assert np.array_equal(predict(model, val_loader, device)[0],
                              predict(reloaded, val_loader, device)[0])
    print("  pipeline: fit -> checkpoint -> reload gives identical predictions")


def check_data_split():
    if not (DATA_DIR / "gtsrb" / "GTSRB" / "Training").is_dir():
        print("  data split: skipped (GTSRB not downloaded yet)")
        return
    from src.data import load_gtsrb, split_train_val

    train_full, _ = load_gtsrb(download=False)
    train_idx, val_idx = split_train_val(train_full)

    def groups(indices):
        return {(train_full._samples[i][1], Path(train_full._samples[i][0]).stem.split("_")[0])
                for i in indices}

    assert not set(train_idx) & set(val_idx)
    assert len(train_idx) + len(val_idx) == len(train_full)
    assert not groups(train_idx) & groups(val_idx), "a track appears in both train and validation"
    assert {c for c, _ in groups(val_idx)} == set(range(NUM_CLASSES))
    assert split_train_val(train_full) == (train_idx, val_idx), "split must be reproducible"
    print(f"  data split: train {len(train_idx)} / val {len(val_idx)}, "
          f"no shared tracks, all classes in validation, reproducible")


def main():
    for check in (check_models, check_stn, check_resnet_freezing, check_metrics,
                  check_training_pipeline, check_data_split):
        print(f"{check.__name__}")
        check()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()

# Lightweight-Traffic-Sign-CNN

Traffic sign recognition with a lightweight CNN that adds a Spatial Transformer Network (STN) and channel attention, built for adverse conditions. Uses the GTSRB dataset (43 classes).

Florida Polytechnic University: Hannah Vutuan and Alexandra Hyatali

## Models

| Name (`--model`) | Model | Role |
|---|---|---|
| `basic_cnn` | Basic CNN | ablation: baseline |
| `cnn_stn` | STN → Basic CNN | ablation: + STN |
| `cnn_attention` | CNN with squeeze-and-excitation channel attention | ablation: + attention |
| `cnn_stn_attention` | STN → CNN with channel attention | **proposed model** |
| `resnet18` | ImageNet-pretrained ResNet-18, fine-tuned | reference baseline |
| HOG + SVM | HOG features + linear SVM (`scripts/train_hog_svm.py`) | classical baseline |

The ablation variants differ only in the component being tested. `cnn_stn` is `basic_cnn` with the STN in front, and `cnn_stn_attention` is `cnn_attention` with the same STN in front. All deep models take 48×48 RGB images in [0, 1]. ResNet-18 resizes and normalises its input internally.

## Repository layout

```
src/
  config.py            shared paths + TrainConfig (same settings for every model)   Hannah
  training.py          training loop, early stopping, checkpoints                   Hannah
  evaluation.py        predictions + metrics on a data loader                       Hannah
  metrics.py           accuracy, macro P/R/F1, per-class scores, confusion matrix   Hannah
  visualization.py     training curves, STN before/after figures                    Hannah
  data/__init__.py     GTSRB download, preprocessing, train/val/test splits         Alex
  models/
    __init__.py        build_model(name): model registry                            Hannah
    basic_cnn.py       BasicCNN                                                     Alex
    cnn_attention.py   ChannelAttention, CNNAttention                               Alex
    stn.py             STN module, STNClassifier                                    Hannah
    cnn_stn.py         CNN + STN                                                    Hannah
    cnn_stn_attention.py  CNN + STN + attention                                     Hannah
    resnet18.py        ResNet-18 fine-tuning baseline                               Hannah
scripts/
  train.py             train any deep model                                         Hannah
  evaluate.py          evaluate a saved checkpoint on val / test                    Hannah
  summarize_results.py comparison table of all runs                                 Hannah
  smoke_test.py        quick checks (no dataset or training needed)                 Hannah
  train_hog_svm.py     HOG + SVM baseline                                           Alex
```

Generated at run time (not committed): `data/`, `checkpoints/`. Also generated: `results/` (metrics JSON) and `figures/` (PNG).

## Data splits

- **Test:** the official GTSRB test set (12,630 images). Deep models are evaluated on it only with `--eval-test`.
- **Train / validation:** torchvision's GTSRB training set (26,640 images), split about 80/20 **by track**: 21,360 train and 5,280 validation images.
  - GTSRB training images come in tracks of 30 consecutive frames of the same physical sign. A whole track goes to either train or validation, never both, so validation scores aren't inflated by near-duplicate frames.
  - The split is stratified by class (all 43 classes appear in both sets) and fixed by the seed in `src/config.py`.
  - HOG + SVM uses the same split.

## Running (Google Colab GPU runtime)

```python
!git clone https://github.com/hvutuan2514/Lightweight-Traffic-Sign-CNN.git
%cd Lightweight-Traffic-Sign-CNN
!pip install -q -r requirements.txt

!python -m scripts.smoke_test                 # ~1 minute sanity check

!python -m scripts.train --model basic_cnn
!python -m scripts.train --model cnn_stn
!python -m scripts.train --model cnn_attention
!python -m scripts.train --model cnn_stn_attention
!python -m scripts.train --model resnet18     # head-only phase, then full fine-tuning
!python -m scripts.train_hog_svm

!python -m scripts.summarize_results          # validation comparison table
```

GTSRB downloads automatically into `data/` on first use. Useful options for `scripts/train.py` include `--epochs`, `--batch-size`, `--lr`, `--patience`, `--run-name`, and `--eval-test`. ResNet-18 also accepts `--head-epochs` and `--finetune-lr`. Run `python -m scripts.train --help` for the full list.

## Outputs

| File | Contents |
|---|---|
| `checkpoints/<run>.pt` | best weights by validation macro-F1, plus model name and config |
| `results/<run>_history.json` | per-epoch train/val loss, accuracy, val macro-F1, learning rate |
| `results/<run>_val_metrics.json` | accuracy, macro precision/recall/F1, per-class scores, confusion matrix |
| `results/<run>_test_metrics.json` | same metrics on the test set (only with `--eval-test`) |
| `figures/<run>_history.png` | training curves |
| `figures/<run>_stn.png` | input images vs. STN output (STN models only) |

## Shared training configuration

These settings come from `TrainConfig` in `src/config.py`:
- AdamW optimizer (lr 1e-3, weight decay 1e-4) with a cosine schedule
- batch size 128, up to 30 epochs
- early stopping after 7 epochs without improvement in validation macro-F1
- gradient clipping at 1.0
- mixed precision on CUDA
- seed 42

ResNet-18 trains only its new head for 3 epochs, then fine-tunes the whole network with lr 1e-4. All other settings are shared.

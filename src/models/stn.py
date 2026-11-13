"""Spatial Transformer Network (Jaderberg et al., NeurIPS 2015).

An STN has three parts:
  1. Localization network - a small CNN that predicts a 2x3 affine matrix theta.
  2. Grid generator       - turns theta into a sampling grid (F.affine_grid).
  3. Sampler              - bilinearly resamples the input on that grid (F.grid_sample).

It lets the classifier undo scale, rotation, translation, and viewing-angle
changes before classification. The last localization layer is initialised to
the identity transform, so training starts from "do nothing" and the STN only
learns to warp when that lowers the loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    """Affine spatial transformer. Input and output shape: (B, C, H, W)."""

    def __init__(self, in_channels=3, img_size=48):
        super().__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            n_features = self.localization(torch.zeros(1, in_channels, img_size, img_size)).numel()

        self.fc_loc = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6),
        )

        # Identity transform at initialisation: theta = [[1, 0, 0], [0, 1, 0]]
        nn.init.zeros_(self.fc_loc[2].weight)
        with torch.no_grad():
            self.fc_loc[2].bias.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def predict_theta(self, x):
        return self.fc_loc(self.localization(x).flatten(1)).view(-1, 2, 3)

    def forward(self, x, return_theta=False):
        theta = self.predict_theta(x)

        # grid_sample is precision-sensitive, so always sample in float32 (even under AMP).
        with torch.autocast(device_type=x.device.type, enabled=False):
            grid = F.affine_grid(theta.float(), list(x.shape), align_corners=False)
            out = F.grid_sample(x.float(), grid, mode="bilinear",
                                padding_mode="border", align_corners=False)
        out = out.to(x.dtype)

        return (out, theta) if return_theta else out


class STNClassifier(nn.Module):
    """x -> STN -> classifier. `classifier` is any nn.Module that returns logits."""

    def __init__(self, stn, classifier):
        super().__init__()
        self.stn = stn
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.stn(x))

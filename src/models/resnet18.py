"""ImageNet-pretrained ResNet-18 (He et al., CVPR 2016), the reference baseline.

Design notes
- Takes the same input as every other model (48x48 RGB in [0, 1]), so it can be
  evaluated on the same clean and corrupted test images. ImageNet mean/std
  normalisation happens inside the model.
- Inputs are bilinearly upsampled to `resize_to` inside the model, because
  ResNet downsamples by 32x and a 48x48 input would end as a 2x2 feature map.
- Fine-tuning has two phases (see scripts/train.py):
    1. freeze_backbone() - train only the new 43-class head;
    2. unfreeze_all()    - fine-tune everything with a smaller learning rate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNet18Classifier(nn.Module):

    def __init__(self, num_classes=43, pretrained=True, dropout=0.2, resize_to=64):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.fc.in_features, num_classes),
        )

        self.resize_to = resize_to
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self._backbone_frozen = False

    def forward(self, x):
        if self.resize_to is not None and x.shape[-2:] != (self.resize_to, self.resize_to):
            x = F.interpolate(x, size=(self.resize_to, self.resize_to),
                              mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.backbone(x)

    def freeze_backbone(self):
        """Phase 1: train only the classification head."""
        for name, p in self.named_parameters():
            p.requires_grad = name.startswith("backbone.fc")
        self._backbone_frozen = True

    def unfreeze_all(self):
        """Phase 2: fine-tune the whole network."""
        for p in self.parameters():
            p.requires_grad = True
        self._backbone_frozen = False

    def train(self, mode=True):
        super().train(mode)
        # While the backbone is frozen, keep its BatchNorm running statistics fixed too.
        if mode and self._backbone_frozen:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

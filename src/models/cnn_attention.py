
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(channels // reduction, 4)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.shape

        weights = self.pool(x).view(batch, channels)
        weights = self.mlp(weights).view(batch, channels, 1, 1)

        return x * weights


class AttentionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=8, pool=True):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ChannelAttention(out_channels, reduction)
        ]

        if pool:
            layers.append(nn.MaxPool2d(2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNNAttention(nn.Module):
    """CNN with channel attention for GTSRB classification."""

    def __init__(self, num_classes=43, dropout=0.25, reduction=8):
        super().__init__()

        self.features = nn.Sequential(
            AttentionBlock(3, 32, reduction),
            AttentionBlock(32, 64, reduction),
            AttentionBlock(64, 128, reduction),

            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            ChannelAttention(128, reduction),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

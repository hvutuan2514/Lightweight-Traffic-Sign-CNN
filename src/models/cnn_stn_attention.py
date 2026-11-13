"""Proposed model: CNN + STN + channel attention.

The same STN used in CNNSTN is placed in front of the CNN with channel
attention (CNNAttention), so this model is exactly "CNN + attention" plus the
STN, and "CNN + STN" plus attention. That keeps the four-way ablation clean:

    BasicCNN  |  CNNSTN  |  CNNAttention  |  CNNSTNAttention
"""

from src.models.cnn_attention import CNNAttention
from src.models.stn import STN, STNClassifier


class CNNSTNAttention(STNClassifier):

    def __init__(self, num_classes=43, img_size=48, in_channels=3, dropout=0.25, reduction=8):
        super().__init__(
            stn=STN(in_channels=in_channels, img_size=img_size),
            classifier=CNNAttention(num_classes=num_classes, dropout=dropout, reduction=reduction),
        )

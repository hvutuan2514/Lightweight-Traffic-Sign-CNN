"""Ablation variant: CNN + STN.

The input goes through the STN and then through exactly the same BasicCNN used
for the "basic CNN" ablation variant, so any change in accuracy or robustness
comes from the STN alone.
"""

from src.models.basic_cnn import BasicCNN
from src.models.stn import STN, STNClassifier


class CNNSTN(STNClassifier):

    def __init__(self, num_classes=43, img_size=48, in_channels=3, dropout=0.25):
        super().__init__(
            stn=STN(in_channels=in_channels, img_size=img_size),
            classifier=BasicCNN(num_classes=num_classes, dropout=dropout),
        )

"""Model registry: every script builds models through build_model() by name."""

from src.config import IMG_SIZE, NUM_CLASSES
from src.models.basic_cnn import BasicCNN
from src.models.cnn_attention import CNNAttention
from src.models.cnn_stn import CNNSTN
from src.models.cnn_stn_attention import CNNSTNAttention
from src.models.resnet18 import ResNet18Classifier
from src.models.stn import STN, STNClassifier

# The four ablation variants, then the reference baseline.
MODEL_NAMES = ("basic_cnn", "cnn_stn", "cnn_attention", "cnn_stn_attention", "resnet18")


def build_model(name, num_classes=NUM_CLASSES, img_size=IMG_SIZE, pretrained=True):
    """Build a model by name. `pretrained` only affects ResNet-18."""
    if name == "basic_cnn":
        return BasicCNN(num_classes=num_classes)
    if name == "cnn_attention":
        return CNNAttention(num_classes=num_classes)
    if name == "cnn_stn":
        return CNNSTN(num_classes=num_classes, img_size=img_size)
    if name == "cnn_stn_attention":
        return CNNSTNAttention(num_classes=num_classes, img_size=img_size)
    if name == "resnet18":
        return ResNet18Classifier(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


__all__ = ["BasicCNN", "CNNAttention", "CNNSTN", "CNNSTNAttention", "ResNet18Classifier",
           "STN", "STNClassifier", "MODEL_NAMES", "build_model", "count_parameters"]

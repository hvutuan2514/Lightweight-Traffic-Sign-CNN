
import json
import joblib
import numpy as np

from pathlib import Path
from torchvision.datasets import GTSRB
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from skimage.feature import hog
from skimage.color import rgb2gray

from src.config import DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR


SEED = 42
IMG_SIZE = 48
NUM_CLASSES = 43


def extract_hog(image_tensor):
    """Extract HOG features from an RGB image."""

    image = image_tensor.permute(1, 2, 0).numpy()
    gray = rgb2gray(image)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    return features


def extract_dataset_features(dataset, indices=None):
    """Extract HOG features and labels from a dataset."""

    if indices is None:
        indices = range(len(dataset))

    features = []
    labels = []

    for i in indices:
        image, label = dataset[i]

        features.append(extract_hog(image))
        labels.append(int(label))

    return np.asarray(features), np.asarray(labels)


def calculate_metrics(y_true, y_pred):
    """Calculate classification performance metrics."""

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=np.arange(NUM_CLASSES)
        ).tolist()
    }


def main():

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    train_dataset = GTSRB(
        root=str(DATA_DIR),
        split="train",
        download=True,
        transform=transform
    )

    test_dataset = GTSRB(
        root=str(DATA_DIR),
        split="test",
        download=True,
        transform=transform
    )

    labels = np.asarray([
        int(label) for _, label in train_dataset._samples
    ])

    indices = np.arange(len(train_dataset))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=SEED,
        stratify=labels
    )

    print("Extracting training HOG features...")
    X_train, y_train = extract_dataset_features(
        train_dataset, train_idx
    )

    print("Extracting validation HOG features...")
    X_val, y_val = extract_dataset_features(
        train_dataset, val_idx
    )

    print("Extracting test HOG features...")
    X_test, y_test = extract_dataset_features(test_dataset)

    print("Training HOG + SVM model...")

    model = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=SEED,
            max_iter=10000
        )
    )

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    val_metrics = calculate_metrics(y_val, val_predictions)
    test_metrics = calculate_metrics(y_test, test_predictions)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "hog_svm_val_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=4)

    with open(RESULTS_DIR / "hog_svm_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    joblib.dump(model, CHECKPOINT_DIR / "hog_svm.joblib")

    print("\nValidation Results:")
    print({k: v for k, v in val_metrics.items()
           if k != "confusion_matrix"})

    print("\nTest Results:")
    print({k: v for k, v in test_metrics.items()
           if k != "confusion_matrix"})

    print("\nHOG + SVM training and evaluation complete!")


if __name__ == "__main__":
    main()

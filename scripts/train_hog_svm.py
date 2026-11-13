"""HOG + linear SVM baseline for GTSRB.

Uses the same images (48x48), the same train/validation split, and the same
metrics as the deep models, so its results are directly comparable.

Run from the repository root:
    python -m scripts.train_hog_svm
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import CHECKPOINT_DIR, IMG_SIZE, NUM_CLASSES, RESULTS_DIR, SEED, VAL_FRACTION
from src.data import load_gtsrb, split_train_val
from src.metrics import compute_metrics, format_metrics, save_metrics


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


def main():

    train_dataset, test_dataset = load_gtsrb(img_size=IMG_SIZE)
    train_idx, val_idx = split_train_val(train_dataset, val_fraction=VAL_FRACTION, seed=SEED)

    print("Extracting training HOG features...")
    X_train, y_train = extract_dataset_features(train_dataset, train_idx)

    print("Extracting validation HOG features...")
    X_val, y_val = extract_dataset_features(train_dataset, val_idx)

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

    val_metrics = compute_metrics(y_val, model.predict(X_val), NUM_CLASSES)
    test_metrics = compute_metrics(y_test, model.predict(X_test), NUM_CLASSES)

    info = dict(run="hog_svm", model="hog_svm", hog_features=int(X_train.shape[1]))
    save_metrics(val_metrics, RESULTS_DIR / "hog_svm_val_metrics.json", split="val", **info)
    save_metrics(test_metrics, RESULTS_DIR / "hog_svm_test_metrics.json", split="test", **info)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CHECKPOINT_DIR / "hog_svm.joblib")

    print("\nValidation Results:")
    print(format_metrics(val_metrics))

    print("\nTest Results:")
    print(format_metrics(test_metrics))

    print("\nHOG + SVM training and evaluation complete!")


if __name__ == "__main__":
    main()

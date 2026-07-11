from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_NAMES, extract_features


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fake Indian note detector.")
    parser.add_argument("--dataset", default="dataset", help="Dataset folder with real/ and fake/ subfolders.")
    parser.add_argument("--denomination", default="500", help="Denomination profile to train against.")
    parser.add_argument("--output", default="models/note_model.joblib", help="Where to save the trained model.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_path = Path(args.output)
    samples, labels, paths = load_dataset(dataset_dir, args.denomination)

    if len(set(labels)) < 2:
        raise SystemExit("Training needs images in both dataset/real and dataset/fake.")
    if len(labels) < 12:
        raise SystemExit("Add at least 12 labelled images total before training. More is much better.")

    x_train, x_test, y_train, y_test, _, test_paths = train_test_split(
        samples,
        labels,
        paths,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=350,
                    class_weight="balanced",
                    min_samples_leaf=2,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "denomination": args.denomination,
            "feature_names": FEATURE_NAMES,
            "labels": ["fake", "real"],
        },
        output_path,
    )

    print(f"Saved model: {output_path}")
    print(f"Training images: {len(y_train)} | Test images: {len(y_test)}")
    print(classification_report(y_test, predictions, target_names=["fake", "real"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Test files:")
    for path, actual, predicted in zip(test_paths, y_test, predictions):
        print(f"- {path} | actual={label_name(actual)} predicted={label_name(predicted)}")


def load_dataset(dataset_dir: Path, denomination: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    vectors = []
    labels = []
    paths = []

    for folder, label in [("fake", 0), ("real", 1)]:
        image_dir = dataset_dir / folder
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue
            result = extract_features(image, denomination)
            vectors.append(result.vector)
            labels.append(label)
            paths.append(str(image_path))

    if not vectors:
        raise SystemExit(f"No images found in {dataset_dir}/real or {dataset_dir}/fake.")

    return np.vstack(vectors), np.array(labels, dtype=np.int64), paths


def label_name(value: int) -> str:
    return "real" if value == 1 else "fake"


if __name__ == "__main__":
    main()

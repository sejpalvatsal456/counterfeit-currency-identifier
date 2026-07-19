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

from src.features import DUAL_FEATURE_NAMES, extract_dual_features


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Expected dataset layout (front/back pairs matched by filename stem):
#
#   dataset/
#     real/
#       front/note001.jpg
#       back/note001.jpg
#     fake/
#       front/note001.jpg
#       back/note001.jpg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fake Indian note detector.")
    parser.add_argument("--dataset", default="dataset", help="Dataset folder with real/ and fake/ subfolders.")
    parser.add_argument("--denomination", default="500", help="Denomination profile to train against.")
    parser.add_argument("--output", default="models/note_model.joblib", help="Where to save the trained model.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_path = Path(args.output)
    samples, labels, pairs = load_dataset(dataset_dir, args.denomination)

    if len(set(labels)) < 2:
        raise SystemExit("Training needs images in both dataset/real and dataset/fake.")
    if len(labels) < 12:
        raise SystemExit("Add at least 12 labelled note pairs total before training. More is much better.")

    x_train, x_test, y_train, y_test, _, test_pairs = train_test_split(
        samples,
        labels,
        pairs,
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
            "feature_names": DUAL_FEATURE_NAMES,
            "labels": ["fake", "real"],
        },
        output_path,
    )

    print(f"Saved model: {output_path}")
    print(f"Training pairs: {len(y_train)} | Test pairs: {len(y_test)}")
    print(classification_report(y_test, predictions, target_names=["fake", "real"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Test files:")
    for (front_path, back_path), actual, predicted in zip(test_pairs, y_test, predictions):
        print(
            f"- front={front_path} back={back_path} "
            f"| actual={label_name(actual)} predicted={label_name(predicted)}"
        )


def load_dataset(
    dataset_dir: Path, denomination: str
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    vectors = []
    labels = []
    pairs: list[tuple[str, str]] = []

    for folder, label in [("fake", 0), ("real", 1)]:
        front_dir = dataset_dir / folder / "front"
        back_dir = dataset_dir / folder / "back"
        if not front_dir.exists() or not back_dir.exists():
            continue

        back_by_stem = {
            path.stem: path for path in back_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
        }

        for front_path in sorted(front_dir.rglob("*")):
            if front_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            back_path = back_by_stem.get(front_path.stem)
            if back_path is None:
                print(f"Skipping {front_path}: no matching back image (expected {folder}/back/{front_path.stem}.*)")
                continue

            front_image = cv2.imread(str(front_path))
            back_image = cv2.imread(str(back_path))
            if front_image is None:
                print(f"Skipping unreadable image: {front_path}")
                continue
            if back_image is None:
                print(f"Skipping unreadable image: {back_path}")
                continue

            result = extract_dual_features(front_image, back_image, denomination)
            vectors.append(result.vector)
            labels.append(label)
            pairs.append((str(front_path), str(back_path)))

    if not vectors:
        raise SystemExit(
            f"No matched front/back image pairs found in {dataset_dir}/real or {dataset_dir}/fake "
            "(expected <label>/front/<name>.jpg and <label>/back/<name>.jpg)."
        )

    return np.vstack(vectors), np.array(labels, dtype=np.int64), pairs


def label_name(value: int) -> str:
    return "real" if value == 1 else "fake"


if __name__ == "__main__":
    main()

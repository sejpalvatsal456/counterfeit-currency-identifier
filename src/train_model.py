from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from tensorflow.keras.applications import MobileNetV2

from src.features import load_training_image


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


INPUT_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30


def load_dataset(dataset_dir: Path):

    images = []
    labels = []

    for folder, label in [
        ("fake", 0),
        ("real", 1),
    ]:

        image_dir = dataset_dir / folder

        if not image_dir.exists():
            continue

        for image_path in image_dir.rglob("*"):

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:

                image = load_training_image(
                    str(image_path)
                )

                images.append(image)
                labels.append(label)

            except Exception as e:

                print(
                    f"Skipping {image_path}: {e}"
                )

    if len(images) == 0:
        raise RuntimeError(
            "Dataset is empty."
        )

    return (
        tf.convert_to_tensor(images),
        tf.convert_to_tensor(labels),
    )


def create_model():

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip(
                "horizontal"
            ),

            layers.RandomRotation(
                0.05
            ),

            layers.RandomZoom(
                0.10
            ),

            layers.RandomContrast(
                0.20
            ),
        ]
    )

    base = MobileNetV2(
        input_shape=(
            INPUT_SIZE,
            INPUT_SIZE,
            3,
        ),
        include_top=False,
        weights="imagenet",
    )

    base.trainable = False

    inputs = tf.keras.Input(
        shape=(
            INPUT_SIZE,
            INPUT_SIZE,
            3,
        )
    )

    x = data_augmentation(inputs)

    x = base(
        x,
        training=False,
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(
        0.30
    )(x)

    x = layers.Dense(
        128,
        activation="relu",
    )(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            1e-3
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )

    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="dataset",
    )

    parser.add_argument(
        "--output",
        default="models/note_model.keras",
    )

    args = parser.parse_args()

    print(
        "Loading dataset..."
    )

    images, labels = load_dataset(
        Path(args.dataset)
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            images,
            labels,
        )
    )

    dataset = dataset.shuffle(
        len(images)
    )

    train_size = int(
        len(images) * 0.8
    )

    train_dataset = (
        dataset
        .take(train_size)
        .batch(BATCH_SIZE)
        .prefetch(
            tf.data.AUTOTUNE
        )
    )

    validation_dataset = (
        dataset
        .skip(train_size)
        .batch(BATCH_SIZE)
        .prefetch(
            tf.data.AUTOTUNE
        )
    )

    model = create_model()

    Path(
        args.output
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),

        ModelCheckpoint(

            filepath=args.output,

            monitor="val_accuracy",

            save_best_only=True,
        ),
    ]

    history = model.fit(

        train_dataset,

        validation_data=validation_dataset,

        epochs=EPOCHS,

        callbacks=callbacks,
    )

    print()

    print(
        "Training finished."
    )

    loss, acc, prec, rec = model.evaluate(
        validation_dataset,
        verbose=0,
    )

    print(
        f"Accuracy : {acc:.4f}"
    )

    print(
        f"Precision: {prec:.4f}"
    )

    print(
        f"Recall   : {rec:.4f}"
    )

    model.save(
        args.output
    )

    print(
        f"Saved model to {args.output}"
    )


if __name__ == "__main__":

    main()
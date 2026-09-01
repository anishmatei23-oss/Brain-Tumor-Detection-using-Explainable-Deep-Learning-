"""Train the brain tumor classifier from the public Kaggle dataset.

The dataset and model outputs are intentionally kept outside Git. Install the
dependencies, configure Kaggle access, and run this file from the repository
root.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import kagglehub
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATASET_HANDLE = "masoudnickparvar/brain-tumor-mri-dataset"
CLASSES = ["glioma", "meningioma", "pituitary"]
IMAGE_SIZE = (224, 224)
SEED = 42


def find_dataset_root(downloaded_path: str | os.PathLike[str]) -> Path:
    """Find the Kaggle dataset directory containing Training and Testing."""
    root = Path(downloaded_path)
    candidates = [root, *root.rglob("*")]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "Training").is_dir() and (candidate / "Testing").is_dir():
            return candidate
    raise FileNotFoundError("The downloaded dataset does not contain Training and Testing folders.")


def download_dataset() -> Path:
    """Download or reuse the public dataset in Kaggle's local cache."""
    print(f"Downloading dataset: {DATASET_HANDLE}")
    return find_dataset_root(kagglehub.dataset_download(DATASET_HANDLE))


def create_generators(dataset_root: Path, batch_size: int):
    train_data = ImageDataGenerator(
        rescale=1 / 255.0,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    evaluation_data = ImageDataGenerator(rescale=1 / 255.0)

    train = train_data.flow_from_directory(
        dataset_root / "Training",
        classes=CLASSES,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED,
    )
    validation = train_data.flow_from_directory(
        dataset_root / "Training",
        classes=CLASSES,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED,
    )
    test = evaluation_data.flow_from_directory(
        dataset_root / "Testing",
        classes=CLASSES,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train, validation, test


def create_model() -> tf.keras.Model:
    base = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    features = base(inputs, training=False)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.Dropout(0.3)(features)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(features)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(epochs: int, batch_size: int, output_path: Path) -> None:
    tf.keras.utils.set_random_seed(SEED)
    dataset_root = download_dataset()
    train_data, validation_data, test_data = create_generators(dataset_root, batch_size)
    print(f"Classes: {train_data.class_indices}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = create_model()
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ModelCheckpoint(output_path, monitor="val_accuracy", save_best_only=True),
    ]
    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=callbacks,
    )
    loss, accuracy = model.evaluate(test_data, verbose=0)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Model saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/models/brain_tumor_cnn.keras"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(arguments.epochs, arguments.batch_size, arguments.output)
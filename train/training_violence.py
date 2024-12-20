# -----------------------------------------------------------------------------
# Author: Yago Boleas, Alberto Sánchez, Guillermo Pérez, Ana Mª Torres
# Project: Bumera
# Date: 17/12/2024
# Description: Python script for training a neural network model to classify 
#              violent and non-violent videos based on human pose data. The 
#              script preprocesses the pose data, generates corresponding images, 
#              and uses transfer learning with various pre-trained models (VGG16, 
#              VGG19, EfficientNetV2, InceptionResNetV2, and ConvNeXt) for 
#              classification. It includes functions for dataset generation, 
#              model training, and saving the trained models and results.
#
# License: This code is released under the MIT License.
#          You are free to use, modify, and distribute this software, provided
#          that proper credit is given to the original authors.
#
# Note: For more details, please refer to the LICENSE file included in the repository.
# -----------------------------------------------------------------------------

import os
import json
import datetime
import numpy as np
import tensorflow as tf
from random import sample
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import (
    vgg16,
    vgg19,
    efficientnet_v2,
    inception_resnet_v2,
    convnext,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

LABELS = (
    "nose",
    "left eye",
    "right eye",
    "left ear",
    "right ear",
    "left shoulder",
    "right shoulder",
    "left elbow",
    "right elbow",
    "left wrist",
    "right wrist",
    "left hip",
    "right hip",
    "left knee",
    "right knee",
    "left ankle",
    "right ankle",
)

CONNECTIONS = [
    ("nose", "left eye"),
    ("left eye", "left ear"),
    ("nose", "right eye"),
    ("right eye", "right ear"),
    ("nose", "left shoulder"),
    ("left shoulder", "left elbow"),
    ("left elbow", "left wrist"),
    ("nose", "right shoulder"),
    ("right shoulder", "right elbow"),
    ("right elbow", "right wrist"),
    ("left shoulder", "left hip"),
    ("right shoulder", "right hip"),
    ("left hip", "right hip"),
    ("left hip", "left knee"),
    ("right hip", "right knee"),
    ("left knee", "left ankle"),
    ("right knee", "right ankle"),
]

BASE_DIR = "poses"


def gen_image(frame: dict) -> Image.Image:
    """
    Generates an image with keypoints and connections drawn based on the frame data.

    :param frame: dict : A dictionary containing keypoints for each pose.
    :return: Image.Image : The generated image with keypoints and connections drawn.
    """
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    draw = ImageDraw.Draw(img)

    if len(frame) > 1:  # Store the frame if there´s more than 1 person
        for pose in frame.keys():
            keypoints = frame[pose]
            for body_part in keypoints.keys():
                point = keypoints[body_part]
                draw.ellipse(
                    (
                        int(point[1]) - 3,
                        int(point[0]) - 3,
                        int(point[1]) + 3,
                        int(point[0]) + 3,
                    ),
                    outline="red",
                    width=1,
                )
            for connection in CONNECTIONS:
                pt1 = keypoints[connection[0]]
                pt2 = keypoints[connection[1]]
                draw.line(
                    (int(pt1[1]), int(pt1[0]), int(pt2[1]), int(pt2[0])),
                    fill=[
                        "orange",
                        "yellow",
                        "lime",
                        "aqua",
                        "blue",
                        "magenta",
                    ][int(int(pose) % 6)],
                    width=3,
                )
    return img

def set_model(name: str, module) -> tf.keras.Model:
    """
    Sets the model by loading it from a file or creating a new one if it doesn't exist.

    :param name: str : The name of the model.
    :param module: The module containing the model architectures.
    :return: tf.keras.Model : The loaded or created model.
    """
    model_path = f"trained_models/{name}"
    if os.path.exists(model_path):
        files = os.listdir(model_path)
        file = max(files, key=lambda file: int(file.split("_")[0]))
        model = tf.keras.models.load_model(f"{model_path}/{file}")
    else:
        os.mkdir(model_path)
        model = create_model(name, module)
    return model

def create_model(name: str, module) -> tf.keras.Model:
    """
    Creates a new model based on the specified name and module.

    :param name: str : The name of the model.
    :param module: The module containing the model architectures.
    :return: tf.keras.Model : The created model.
    """
    match name:
        case "vgg16":
            base_model = module.VGG16(
                weights="imagenet", include_top=False, input_shape=img_shape
            )
        case "vgg19":
            base_model = module.VGG19(
                weights="imagenet", include_top=False, input_shape=img_shape
            )
        case "efficientnet_v2":
            base_model = module.EfficientNetV2M(
                weights="imagenet", include_top=False, input_shape=img_shape
            )
        case "inception_resnet_v2":
            base_model = module.InceptionResNetV2(
                weights="imagenet", include_top=False, input_shape=img_shape
            )
        case "convnext":
            base_model = module.ConvNeXtLarge(
                weights="imagenet", include_top=False, input_shape=img_shape
            )
    base_model.trainable = False
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def generate_dataset(files: list, labels: list, dir1: str, dir2: str):
    """
    Generates a dataset from video files.

    :param files: list : List of video files.
    :param labels: list : List of labels corresponding to the video files.
    :param dir1: str : Directory for one class of videos.
    :param dir2: str : Directory for the other class of videos.
    :yield: tuple : A tuple containing the preprocessed image and its label.
    """
    for i, video in enumerate(files):
        path = os.path.join(dir1 if labels[i] else dir2, video)
        with open(path, "rt") as file:
            poses = json.load(file)
        num_frames = len(poses)
        selected_frames = sorted(sample(range(num_frames), min(20, num_frames)))
        for frame_index in selected_frames:
            frame = poses[frame_index]
            if len(frame) > 1:
                img = gen_image(frame)
                yield module.preprocess_input(np.array(img)), labels[i]

def set_dataset(module, data: list, labels: list, train_ds: bool = False) -> tf.data.Dataset:
    """
    Sets up the dataset for training or validation.

    :param module: The module containing the preprocessing function.
    :param data: list : List of data files.
    :param labels: list : List of labels corresponding to the data files.
    :param train_ds: bool : Flag indicating if the dataset is for training.
    :return: tf.data.Dataset : The configured dataset.
    """
    dataset = (
        tf.data.Dataset.from_generator(
            generate_dataset,
            args=(data, labels, violence_dir, nonviolence_dir),
            output_signature=(
                tf.TensorSpec(shape=img_shape, dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )
        .batch(batch_size)
        .prefetch(batch_size)
    )
    return dataset.shuffle(batch_size * 2).repeat() if train_ds else dataset

def save_model(dir: str, model: tf.keras.Model, train_summary: dict):
    """
    Saves the trained model to a file.

    :param dir: str : The directory to save the model.
    :param model: tf.keras.Model : The trained model to save.
    :param train_summary: dict : The training summary containing accuracy metrics.
    """
    now = datetime.datetime.now()
    acc = train_summary["accuracy"][-1]
    val_acc = train_summary["val_accuracy"][-1]
    filename = f"{now.day:02}{now.hour}{now.minute}_{int(acc*100)}_{int(val_acc*100)}"
    model.save(f"trained_models/{dir}/{filename}")

def save_results(model: str, train_summary: dict, time: datetime.timedelta):
    """
    Saves the training results to a JSON file.

    :param model: str : The name of the model.
    :param train_summary: dict : The training summary containing metrics.
    :param time: datetime.timedelta : The duration of the training.
    """
    if os.path.exists("trained_models/results.json"):
        with open("trained_models/results.json", "rt") as file:
            results = json.load(file)
    else:
        structure = {
            "duration_train": [],
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        results = dict()
        for mod in models:
            results[mod] = structure
    results[model]["duration_train"].append(
        f"{time.seconds // 3600}h {(time.seconds // 60) % 60}min {time.seconds % 60}s"
    )
    results[model]["loss"] += train_summary["loss"]
    results[model]["accuracy"] += train_summary["accuracy"]
    results[model]["val_loss"] += train_summary["val_loss"]
    results[model]["val_accuracy"] += train_summary["val_accuracy"]
    with open("trained_models/results.json", "w") as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    # Data directories definition
    violence_dir = os.path.join(BASE_DIR, "Violence")
    nonviolence_dir = os.path.join(BASE_DIR, "NonViolence")

    violence_videos = os.listdir(violence_dir)
    labels_violence = [1] * len(violence_videos)
    nonviolence_videos = os.listdir(nonviolence_dir)
    labels_nonviolence = [0] * len(nonviolence_videos)
    labels = labels_violence + labels_nonviolence
    train_val, test, train_val_labels, testing_labels = train_test_split(
        violence_videos + nonviolence_videos,
        labels,
        test_size=0.1,
        stratify=labels,
        shuffle=True,
    )
    train, validate, fit_labels, val_labels = train_test_split(
        train_val,
        train_val_labels,
        test_size=0.2,
        stratify=train_val_labels,
        shuffle=True,
    )

    # General variables
    img_shape = (224, 224, 3)
    dropout_rate = 0.3
    batch_size = 32
    num_epochs = 10
    steps_epoch = 500
    models = ("vgg16", "vgg19", "efficientnet_v2", "inception_resnet_v2", "convnext")
    modules = dict(
        zip(models, [vgg16, vgg19, efficientnet_v2, inception_resnet_v2, convnext])
    )

    # Training Pipeline
    # while True:
    for name, module in modules.items():
        # Model load
        model = set_model(name, module)
        # Dataset generation
        train_dataset = set_dataset(module, train, fit_labels, True)
        validate_dataset = set_dataset(module, validate, val_labels)
        test_dataset = set_dataset(module, test, testing_labels)
        # Model training
        start = datetime.datetime.now()
        history = model.fit(
            train_dataset,
            batch_size=batch_size,
            epochs=num_epochs,
            steps_per_epoch=steps_epoch,
            validation_data=validate_dataset,
        )
        end = datetime.datetime.now()

        # Results storing
        save_model(name, model, history.history)
        save_results(name, history.history, end - start)

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from PIL import Image

import lib_models
import lib_scenario
from lib_image import normalize_image

WATERBIRDS_TARBALL_LINK = (
    "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
)


def transform_image(image: np.ndarray) -> tf.Tensor:
    image = image[np.newaxis, :, :, :]
    image = tf.cast(image / 255, "float32")
    initial_resolution = (256, 256)
    image = tf.keras.layers.experimental.preprocessing.Resizing((*initial_resolution))(
        image
    )
    image = tf.keras.layers.experimental.preprocessing.CenterCrop(224, 224)(image)

    image = normalize_image(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return image


def read_image(filepath: Path):
    return np.asarray(Image.open(str(filepath)).convert("RGB"))


def download_waterbirds(cache_dir: Path):
    print("Downloading tarball")
    tf.keras.utils.get_file(
        "waterbirds", WATERBIRDS_TARBALL_LINK, extract=True, cache_dir=str(cache_dir)
    )


# y: | 0 -> land bird
#    | 1 -> water bird
#
# y_biased: | 0 -> land
#           | 1 -> water
def generate_waterbirds(cache_dir: Path, split: str):
    dataset_dir = cache_dir / "datasets" / "waterbird_complete95_forest2water2"
    if not dataset_dir.exists():
        download_waterbirds(cache_dir)

    dfm = pd.read_csv(dataset_dir / "metadata.csv")
    split_dict = {"train": 0, "val": 1, "test": 2}
    dfm = dfm[dfm["split"] == split_dict[split]].reset_index(drop=True)
    X = [
        transform_image(read_image(dataset_dir / f)) for f in dfm["img_filename"].values
    ]
    X = tf.concat(X, 0)
    y = dfm["y"].values
    y_biased = dfm["place"].values

    return X, y, y_biased


def load_waterbirds(cache_dir: Path, split: str, include_biased: bool = False):
    cache_filename = cache_dir / ".datasets" / f"waterbirds_processed_{split}.npz"

    if not cache_filename.exists():
        print(f"Generating waterbirds and saving to {cache_filename}")
        X, y, y_biased = generate_waterbirds(cache_dir, split)
        np.savez(cache_filename, X=X, y=y, y_biased=y_biased)

    npz = np.load(cache_filename)

    X, y, y_biased = npz["X"], npz["y"], npz["y_biased"]

    if include_biased:
        return tf.data.Dataset.from_tensor_slices((X, y, y_biased))
    else:
        return tf.data.Dataset.from_tensor_slices((X, y))


class WaterbirdsScenario(lib_scenario.Scenario):
    def __init__(self):
        self.cache_dir = Path(os.environ["SCRATCH"])

    def generate_training_data(self) -> tf.data.Dataset:
        return load_waterbirds(self.cache_dir, "train", False).cache()

    def generate_id_testing_data(self) -> tf.data.Dataset:
        D = load_waterbirds(self.cache_dir, "test", True)

        def filter_for_id(X, y, y_biased):
            return y == y_biased

        def discard_biased(X, y, y_biased):
            return X, y

        return D.filter(filter_for_id).map(discard_biased).cache()

    def generate_ood_testing_data(self) -> tf.data.Dataset:
        D = load_waterbirds(self.cache_dir, "test", True)

        def filter_for_ood(X, y, y_biased):
            return y != y_biased

        def discard_biased(X, y, y_biased):
            return X, y

        return D.filter(filter_for_ood).map(discard_biased).cache()

    def get_num_classes(self):
        return 2

    def get_input_size(self):
        return (224, 224, 3)

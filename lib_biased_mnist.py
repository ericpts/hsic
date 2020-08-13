"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

Python implementation of Biased-MNIST.
Courtesy of
https://github.com/clovaai/rebias/blob/master/datasets/colour_mnist.py
"""
import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import gin
import gin.tf
import argparse
import yaml
from datetime import datetime
from pathlib import Path
import json
from typing import List, Tuple, Dict
import lib_problem


class ColourBiasedMNIST(MNIST):
    """
    We manually select ten colours to synthetic colour bias. (See `COLOUR_MAP` for the colour configuration)
    Usage is exactly same as torchvision MNIST dataset class.

    You have two paramters to control the level of bias.

    Parameters
    ----------
    root : str
        path to MNIST dataset.
    data_label_correlation : float, default=1.0
        Here, each class has the pre-defined colour (bias).
        data_label_correlation, or `rho` controls the level of the dataset bias.

        A sample is coloured with
            - the pre-defined colour with probability `rho`,
            - coloured with one of the other colours with probability `1 - rho`.
              The number of ``other colours'' is controlled by `n_confusing_labels` (default: 9).
        Note that the colour is injected into the background of the image (see `_binary_to_colour`).

        Hence, we have
            - Perfectly biased dataset with rho=1.0
            - Perfectly unbiased with rho=0.1 (1/10) ==> our ``unbiased'' setting in the test time.
        In the paper, we explore the high correlations but with small hints, e.g., rho=0.999.

    n_confusing_labels : int, default=9
        In the real-world cases, biases are not equally distributed, but highly unbalanced.
        We mimic the unbalanced biases by changing the number of confusing colours for each class.
        In the paper, we use n_confusing_labels=9, i.e., during training, the model can observe
        all colours for each class. However, you can make the problem harder by setting smaller n_confusing_labels, e.g., 2.
        We suggest to researchers considering this benchmark for future researches.
    """

    COLOUR_MAP = [
        # 0
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [225, 225, 0],
        [225, 0, 225],
        # 5
        [0, 255, 255],
        [255, 128, 0],
        [255, 0, 128],
        [128, 0, 255],
        [128, 128, 128],
    ]

    def __init__(
        self,
        root,
        train=True,
        download=False,
        data_label_correlation=1.0,
        n_confusing_labels=9,
        background_noise_level=0,
    ):
        super().__init__(
            root, train=train, download=download,
        )
        np.random.seed(0)
        self.background_noise_level = background_noise_level

        self.data_label_correlation = data_label_correlation
        self.n_confusing_labels = n_confusing_labels
        data, targets, biased_targets = self.build_biased_mnist()

        indices = np.arange(len(self.data))
        self._shuffle(indices)

        self.data = data[indices].numpy().astype(np.float32)
        self.targets = targets[indices].numpy().astype(np.int32)
        self.biased_targets = biased_targets[indices].numpy().astype(np.int32)

        self.data /= 255.0

    @property
    def raw_folder(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, "processed")

    def _shuffle(self, iteratable):
        np.random.shuffle(iteratable)

    def _binary_to_colour(self, data, colour):
        fg_data = torch.zeros_like(data)
        fg_data[data != 0] = 255
        fg_data[data == 0] = 0
        fg_data = torch.stack([fg_data, fg_data, fg_data], dim=-1)

        bg_data = torch.zeros_like(data)
        bg_data[data == 0] = 1
        bg_data[data != 0] = 0
        bg_data = torch.stack([bg_data, bg_data, bg_data], dim=3)
        bg_data = bg_data * torch.ByteTensor(colour)

        noise = torch.randint(
            -self.background_noise_level,
            self.background_noise_level + 1,
            bg_data.size(),
        )
        noise[data != 0] = 0
        bg_data = torch.clamp(bg_data + noise, 0, 255).type(fg_data.dtype)
        data = fg_data + bg_data
        return data

    def _make_biased_mnist(self, indices, label):
        return (
            self._binary_to_colour(self.data[indices], self.COLOUR_MAP[label]),
            self.targets[indices],
        )

    def _update_bias_indices(self, bias_indices, label):
        if self.n_confusing_labels > 9 or self.n_confusing_labels < 1:
            raise ValueError(self.n_confusing_labels)

        indices = np.where((self.targets == label).numpy())[0]
        self._shuffle(indices)
        indices = torch.LongTensor(indices)

        n_samples = len(indices)
        n_correlated_samples = int(n_samples * self.data_label_correlation)
        n_decorrelated_per_class = int(
            np.ceil((n_samples - n_correlated_samples) / (self.n_confusing_labels))
        )

        correlated_indices = indices[:n_correlated_samples]
        bias_indices[label] = torch.cat([bias_indices[label], correlated_indices])

        decorrelated_indices = torch.split(
            indices[n_correlated_samples:], n_decorrelated_per_class
        )

        other_labels = [
            _label % 10
            for _label in range(label + 1, label + 1 + self.n_confusing_labels)
        ]
        self._shuffle(other_labels)

        for idx, _indices in enumerate(decorrelated_indices):
            _label = other_labels[idx]
            bias_indices[_label] = torch.cat([bias_indices[_label], _indices])

    def build_biased_mnist(self):
        """Build biased MNIST.
        """
        n_labels = self.targets.max().item() + 1

        bias_indices = {label: torch.LongTensor() for label in range(n_labels)}
        for label in range(n_labels):
            self._update_bias_indices(bias_indices, label)

        data = torch.ByteTensor()
        targets = torch.LongTensor()
        biased_targets = []

        for bias_label, indices in bias_indices.items():
            _data, _targets = self._make_biased_mnist(indices, bias_label)
            data = torch.cat([data, _data])
            targets = torch.cat([targets, _targets])
            biased_targets.extend([bias_label] * len(indices))

        biased_targets = torch.LongTensor(biased_targets)
        return data, targets, biased_targets

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        return img, target, int(self.biased_targets[index])


def get_biased_mnist_data(
    root: Path,
    data_label_correlation: float,
    train: bool = True,
    n_confusing_labels: int = 9,
    force_regenerate: bool = False,
    background_noise_level: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    train_ext = "train" if train else "test"
    fname = (
        Path(root).expanduser()
        / f"data_{data_label_correlation}_{n_confusing_labels}_{train_ext}_{background_noise_level}.npz"
    )

    if force_regenerate or not fname.exists():
        dataset = ColourBiasedMNIST(
            root,
            train=train,
            download=True,
            data_label_correlation=data_label_correlation,
            n_confusing_labels=n_confusing_labels,
            background_noise_level=background_noise_level,
        )
        img, labels, biased_labels = (
            dataset.data,
            dataset.targets,
            dataset.biased_targets,
        )
        np.savez(fname, img=img, labels=labels, biased_labels=biased_labels)

    npz = np.load(fname)
    return (npz["img"], npz["labels"], npz["biased_labels"])


@gin.configurable
def get_weight_regularizer(strength: float = 0.01):
    return tf.keras.regularizers.l2(strength)


def make_base_cnn_model(n_classes: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input((28, 28, 3))

    X = inputs
    X = tf.keras.layers.Conv2D(4, kernel_size=3)(X)
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.BatchNormalization()(X)

    X = tf.keras.layers.Conv2D(8, kernel_size=3)(X)
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.BatchNormalization()(X)

    X = tf.keras.layers.Conv2D(16, kernel_size=3)(X)
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.BatchNormalization()(X)

    X = tf.keras.layers.GlobalAveragePooling2D()(X)

    feature_extractor = X

    X = tf.keras.layers.Dense(n_classes)(X)
    return tf.keras.Model(inputs, outputs=[feature_extractor, X])


def make_base_mlp_model(n_classes: int) -> tf.keras.Model:
    reg = get_weight_regularizer()
    inputs = tf.keras.layers.Input((28, 28, 3))
    X = inputs
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(n_classes * 2, kernel_regularizer=reg)(X)
    feature_extractor = X
    X = tf.keras.layers.Dense(n_classes, kernel_regularizer=reg)(X)
    return tf.keras.Model(inputs, outputs=[feature_extractor, X])


@gin.configurable
class BiasedMnistProblem(lib_problem.Problem):
    def __init__(
        self,
        training_data_label_correlation: float = 0.99,
        filter_for_digits: List[int] = list(range(10)),
        model_type: str = "mlp",
        background_noise_level: int = 0,
        *args,
        **kwargs,
    ) -> None:
        if model_type == "mlp":
            make_base_model = lambda: make_base_mlp_model(len(filter_for_digits))
        elif model_type == "cnn":
            make_base_model = lambda: make_base_cnn_model(len(filter_for_digits))
        else:
            raise ValueError(f"Unknown model_type: {model_type}!")
        super().__init__("biased_mnist_problem", make_base_model, *args, **kwargs)
        self.training_data_label_correlation = training_data_label_correlation
        self.filter_for_digits = tf.convert_to_tensor(filter_for_digits)
        self.background_noise_level = background_noise_level

    def filter_tensors(
        self, X: tf.Tensor, y: tf.Tensor, y_biased: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] == y_biased.shape[0]
        matches_any = [tf.math.equal(y, d) for d in self.filter_for_digits]
        select = tf.math.reduce_any(matches_any, axis=0)
        return X[select], y[select], y_biased[select]

    def generate_training_data(self, include_bias: bool = False) -> tf.data.Dataset:
        if include_bias:
            to_select = 3
        else:
            to_select = 2
        return (
            tf.data.Dataset.from_tensor_slices(
                self.filter_tensors(
                    *get_biased_mnist_data(
                        "~/.datasets/mnist/",
                        self.training_data_label_correlation,
                        train=True,
                        background_noise_level=self.background_noise_level,
                    )
                )[:to_select]
            )
            .cache()
            .shuffle(60_000)
        )

    def generate_testing_data(self, include_bias: bool = False) -> tf.data.Dataset:
        if include_bias:
            to_select = 3
        else:
            to_select = 2
        return tf.data.Dataset.from_tensor_slices(
            self.filter_tensors(
                *get_biased_mnist_data(
                    "~/.datasets/mnist/", 0.1, train=False, background_noise_level=0,
                )
            )[:to_select]
        ).cache()


def regenerate_all_data(
    label_correlations: List[float], background_noise_levels: List[int]
):
    get_biased_mnist_data("~/.datasets/mnist/", 0.1, train=False, force_regenerate=True)

    for bg_noise in background_noise_levels:
        for label_corr in label_correlations:
            get_biased_mnist_data(
                "~/.datasets/mnist/",
                label_corr,
                train=True,
                force_regenerate=True,
                background_noise_level=bg_noise,
            )


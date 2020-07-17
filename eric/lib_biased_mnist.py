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


class BiasedMNIST(MNIST):
    """A base class for Biased-MNIST.
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
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [225, 225, 0],
        [225, 0, 225],
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
        transform=None,
        target_transform=None,
        download=False,
        data_label_correlation=1.0,
        n_confusing_labels=9,
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.n_shuffles = 0
        self.random = True

        self.data_label_correlation = data_label_correlation
        self.n_confusing_labels = n_confusing_labels
        self.data, self.targets, self.biased_targets = self.build_biased_mnist()

        indices = np.arange(len(self.data))
        self._shuffle(indices)

        self.data = self.data[indices].numpy()
        self.targets = self.targets[indices]
        self.biased_targets = self.biased_targets[indices]

    @property
    def raw_folder(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, "processed")

    def _shuffle(self, iteratable):
        if self.random:
            np.random.seed(self.n_shuffles)
            self.n_shuffles += 1
            np.random.shuffle(iteratable)

    def _make_biased_mnist(self, indices, label):
        raise NotImplementedError

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
        img = Image.fromarray(img.astype(np.uint8), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, int(self.biased_targets[index])


class ColourBiasedMNIST(BiasedMNIST):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        data_label_correlation=1.0,
        n_confusing_labels=9,
    ):
        super(ColourBiasedMNIST, self).__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            data_label_correlation=data_label_correlation,
            n_confusing_labels=n_confusing_labels,
        )

    def _binary_to_colour(self, data, colour):
        fg_data = torch.zeros_like(data)
        fg_data[data != 0] = 255
        fg_data[data == 0] = 0
        fg_data = torch.stack([fg_data, fg_data, fg_data], dim=1)

        bg_data = torch.zeros_like(data)
        bg_data[data == 0] = 1
        bg_data[data != 0] = 0
        bg_data = torch.stack([bg_data, bg_data, bg_data], dim=3)
        bg_data = bg_data * torch.ByteTensor(colour)
        bg_data = bg_data.permute(0, 3, 1, 2)

        data = fg_data + bg_data
        return data.permute(0, 2, 3, 1)

    def _make_biased_mnist(self, indices, label):
        return (
            self._binary_to_colour(self.data[indices], self.COLOUR_MAP[label]),
            self.targets[indices],
        )


def make_biased_mnist_data(root, data_label_correlation, train=True):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    dataset = ColourBiasedMNIST(
        root,
        train=train,
        transform=transform,
        download=True,
        data_label_correlation=data_label_correlation,
        n_confusing_labels=9,
    )

    def gen():
        for img, target, biased_target in dataset:
            x = img.numpy()
            # x = np.moveaxis(x, 0, 2)
            yield x, target  # , biased_target

        return None

    return gen


def make_base_mlp_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input((3, 28, 28))
    X = inputs
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(11)(X)
    feature_extractor = X
    X = tf.keras.layers.Dense(10)(X)
    return tf.keras.Model(inputs, outputs=[feature_extractor, X])


def make_base_cnn_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input((28, 28, 3))
    X = inputs
    X = tf.keras.layers.Conv2D(3, 5, kernel_regularizer=tf.keras.regularizers.L2(1e-4))(
        X
    )
    X = tf.keras.layers.ReLU()(X)

    X = tf.keras.layers.Conv2D(6, 5, kernel_regularizer=tf.keras.regularizers.L2(1e-4))(
        X
    )
    X = tf.keras.layers.ReLU()(X)

    X = tf.keras.layers.Conv2D(
        12, 5, kernel_regularizer=tf.keras.regularizers.L2(1e-4)
    )(X)
    X = tf.keras.layers.ReLU()(X)

    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(
        10, kernel_regularizer=tf.keras.regularizers.L2(1e-4), activation="relu"
    )(X)
    feature_extractor = X
    X = tf.keras.layers.Dense(
        10, kernel_regularizer=tf.keras.regularizers.L2(1e-4), activation="linear"
    )(X)

    return tf.keras.Model(inputs, outputs=[feature_extractor, X])


@gin.configurable
class BiasedMnistProblem(lib_problem.Problem):
    def __init__(
        self,
        training_data_label_correlation: float = 0.99,
        base_model: str = "mlp",
        *args,
        **kwargs,
    ) -> None:
        make_base_model = None
        if base_model == "cnn":
            make_base_model = make_base_cnn_model
        elif base_model == "mlp":
            make_base_model = make_base_mlp_model
        else:
            raise ValueError(
                f"Unrecognized base model: {base_model}; expected one of cnn, mlp"
            )
        super().__init__("biased_mnist_problem", make_base_model, *args, **kwargs)
        self.training_data_label_correlation = training_data_label_correlation

    def generate_training_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        return (
            tf.data.Dataset.from_generator(
                make_biased_mnist_data(
                    "~/.datasets/mnist/",
                    self.training_data_label_correlation,
                    train=True,
                ),
                output_types=(tf.float32, tf.int64),
            )
            .cache()
            .shuffle(60_000)
        )

    def generate_testing_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        return tf.data.Dataset.from_generator(
            make_biased_mnist_data("~/.datasets/mnist/", 0.0, train=False),
            output_types=(tf.float32, tf.int64),
        ).cache()


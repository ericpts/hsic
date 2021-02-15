"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

Python implementation of Biased-MNIST.
Courtesy of
https://github.com/clovaai/rebias/blob/master/datasets/colour_mnist.py
"""
import os
import torch
from torchvision.datasets import MNIST
import numpy as np
import tensorflow as tf
import gin
import gin.tf
from pathlib import Path
from typing import List, Tuple, Union, Literal
import lib_scenario


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

    ID_COLOR_MAP = [
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

    OOD_COLOR_MAP = [
        # 0
        [128, 0, 0],
        [0, 128, 0],
        [0, 0, 128],
        [100, 100, 0],
        [100, 0, 100],
        # 5
        [0, 128, 128],
        [128, 64, 0],
        [128, 0, 64],
        [64, 0, 128],
        [64, 64, 64],
    ]

    def __init__(
        self,
        root,
        color_map: Union[Literal["id"], Literal["ood"]],
        train=True,
        download=False,
        data_label_correlation=1.0,
        n_confusing_labels=9,
        background_noise_level=0,
    ):
        super().__init__(
            root,
            train=train,
            download=download,
        )
        np.random.seed(0)
        self.color_map = color_map
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

        noise = torch.randint(
            -self.background_noise_level,
            self.background_noise_level + 1,
            (data.shape[0], 3),
        )
        noisy_colour = torch.clamp(
            noise + np.asarray(colour)[np.newaxis, :], 0, 255
        ).type(fg_data.dtype)

        bg_data = torch.zeros_like(data)
        bg_data[data == 0] = 1
        bg_data[data != 0] = 0

        bg_data = torch.stack([bg_data, bg_data, bg_data], dim=3)
        bg_data = bg_data * torch.ByteTensor(noisy_colour[:, np.newaxis, np.newaxis, :])

        data = fg_data + bg_data
        return data

    def _make_biased_mnist(self, indices, label):
        if self.color_map == "id":
            colour_map = self.ID_COLOR_MAP
        else:
            assert self.color_map == "ood"
            colour_map = self.OOD_COLOR_MAP
        return (
            self._binary_to_colour(self.data[indices], colour_map[label]),
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
        """Build biased MNIST."""
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
    color_map: Union[Literal["id"], Literal["ood"]],
    train: bool = True,
    n_confusing_labels: int = 9,
    background_noise_level: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = ColourBiasedMNIST(
        root,
        color_map=color_map,
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
    return img, labels, biased_labels


@gin.configurable
class BiasedMnistScenario(lib_scenario.Scenario):
    def __init__(
        self,
        training_data_label_correlation: float = 0.99,
        filter_for_digits: List[int] = list(range(10)),
        background_noise_level: int = 0,
    ) -> None:
        self.training_data_label_correlation = training_data_label_correlation
        self.filter_for_digits = tf.convert_to_tensor(sorted(filter_for_digits))
        self.background_noise_level = background_noise_level

        self.cache_path = Path(os.environ["SCRATCH"]) / ".datasets" / "mnist"

    def filter_tensors(
        self, X: tf.Tensor, y: tf.Tensor, y_biased: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
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
                        self.cache_path,
                        self.training_data_label_correlation,
                        "id",
                        train=True,
                        background_noise_level=self.background_noise_level,
                    )
                )[:to_select]
            )
            .cache()
            .shuffle(60_000)
        )

    def generate_ood_testing_data(self, include_bias: bool = False) -> tf.data.Dataset:
        if include_bias:
            to_select = 3
        else:
            to_select = 2
        return tf.data.Dataset.from_tensor_slices(
            self.filter_tensors(
                *get_biased_mnist_data(
                    self.cache_path,
                    0.1,
                    "ood",
                    train=False,
                    background_noise_level=self.background_noise_level,
                )
            )[:to_select]
        ).cache()

    def generate_id_testing_data(self, include_bias: bool = False) -> tf.data.Dataset:
        if include_bias:
            to_select = 3
        else:
            to_select = 2
        return tf.data.Dataset.from_tensor_slices(
            self.filter_tensors(
                *get_biased_mnist_data(
                    self.cache_path,
                    self.training_data_label_correlation,
                    "id",
                    train=False,
                    background_noise_level=self.background_noise_level,
                )
            )[:to_select]
        ).cache()

    def get_num_classes(self):
        return 10

    def get_input_size(self):
        return (28, 28, 3)

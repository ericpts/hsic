import numpy as np
import tensorflow as tf
from typing import Tuple
import lib_problem


def make_base_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(2)
    X = inputs
    X = tf.keras.layers.Dense(1, activation="linear", use_bias=False)(X)
    feature_extractor = X
    X = tf.keras.layers.Dense(1, activation="linear", use_bias=True)(X)
    return tf.keras.Model(inputs, outputs=[feature_extractor, X])


class ToyProblem(lib_problem.Problem):
    def __init__(self) -> None:
        super().__init__("toy_quadrant_problem", make_base_model)

    def generate_training_data(
        self, n_samples: int = 100_000, sigma_c: float = 0.4, sigma_s: float = 0.4
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        np.random.seed(0)

        y = np.random.choice([1, -1], size=(n_samples, 1))

        c = np.random.normal(2 * y, scale=sigma_c, size=(n_samples, 1))
        s = np.random.normal(2 * y, scale=sigma_s, size=(n_samples, 1))

        X = np.concatenate([c, s], axis=-1)

        y = tf.reshape(tf.where(y > 0, x=1, y=0), (-1,))
        return tf.data.Dataset.from_tensor_slices((X, tf.cast(y, tf.int32))).shuffle(
            n_samples
        )

    def generate_testing_data(
        self, n_samples: int = 20_000, sigma_c: float = 0.4, sigma_s: float = 0.4
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        np.random.seed(0)

        y = np.random.choice([1, -1], size=(n_samples, 1))
        c = np.random.normal(2 * y, scale=sigma_c, size=(n_samples, 1))

        z = np.random.choice([1, -1], size=(n_samples, 1))
        s = np.random.normal(2 * z, scale=sigma_s, size=(n_samples, 1))

        X = np.concatenate([c, s], axis=-1)

        y = tf.reshape(tf.where(y > 0, x=1, y=0), (-1,))
        return tf.data.Dataset.from_tensor_slices((X, tf.cast(y, tf.int32)))

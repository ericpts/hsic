import gin
import numpy as np
import tensorflow as tf
from typing import Tuple
import lib_scenario


@gin.configurable
class ToyScenario(lib_scenario.Scenario):
    def __init__(
        self, n_samples: int = 100_000, sigma_c: float = 0.4, sigma_s: float = 0.4
    ) -> None:
        self.n_samples = n_samples
        self.sigma_c = sigma_c
        self.sigma_s = sigma_s

    def generate_correlated(self):
        y = np.random.choice([1, -1], size=(self.n_samples, 1))

        c = np.random.normal(2 * y, scale=self.sigma_c, size=(self.n_samples, 1))
        s = np.random.normal(2 * y, scale=self.sigma_s, size=(self.n_samples, 1))

        X = np.concatenate([c, s], axis=-1)

        y = tf.reshape(tf.where(y > 0, x=1, y=0), (-1,))
        return tf.data.Dataset.from_tensor_slices((X, tf.cast(y, tf.int32))).shuffle(
            self.n_samples
        )

    def generate_uncorrelated(self):
        np.random.seed(0)

        y = np.random.choice([1, -1], size=(self.n_samples, 1))
        c = np.random.normal(2 * y, scale=self.sigma_c, size=(self.n_samples, 1))

        z = np.random.choice([1, -1], size=(self.n_samples, 1))
        s = np.random.normal(2 * z, scale=self.sigma_s, size=(self.n_samples, 1))

        X = np.concatenate([c, s], axis=-1)

        y = tf.reshape(tf.where(y > 0, x=1, y=0), (-1,))
        return tf.data.Dataset.from_tensor_slices((X, tf.cast(y, tf.int32)))

    def generate_training_data(self) -> tf.data.Dataset:
        np.random.seed(0)
        return self.generate_correlated()

    def generate_ood_testing_data(self) -> tf.data.Dataset:
        np.random.seed(1)
        return self.generate_uncorrelated()

    def generate_id_testing_data(self) -> tf.data.Dataset:
        np.random.seed(2)
        return self.generate_correlated()

    def get_num_classes(self):
        return 2

    def get_input_size(self):
        return (2,)

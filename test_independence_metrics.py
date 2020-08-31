import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import unittest
from lib_independence_metrics import *
from typing import Dict, Callable
from lib_util import with_grid_values


def large_grid() -> Dict:
    return {
        "stddev": [0.1, 0.5, 1.0, 5.0],
        "shape": [
            (5, 1),
            (10, 1),
            (100, 1),
            (1000, 1),
            (5, 5),
            (10, 5),
            (100, 5),
            (1000, 5),
            (5, 50),
            (10, 50),
            (100, 50),
            (1000, 50),
            (5, 500),
            (10, 500),
            (100, 500),
            (1000, 500),
        ],
    }


class IndependenceMetricsTest(unittest.TestCase):
    def test_dhsic(self):
        def once(stddev, shape):
            y = np.random.choice([1, -1], shape)
            c = y + np.random.normal(0, stddev, y.shape)
            s = y + np.random.normal(0, stddev, y.shape)
            c = tf.cast(c, "float32")
            s = tf.cast(s, "float32")
            k = tfp.math.psd_kernels.ExponentiatedQuadratic()

            with self.subTest(stddev=stddev, shape=shape):
                self.assert_tensors_almost_equal(hsic([c, s], k), dhsic([c, s], k))

        with_grid_values(large_grid(), once)

    def test_chsic(self):
        def once(stddev, N):
            y = np.random.choice([1, 0], N)
            c = y + np.random.normal(0, stddev, (N, 1))
            s = y + np.random.normal(0, stddev, (N, 1))

            c = tf.cast(c, "float32")
            s = tf.cast(s, "float32")

            k = tfp.math.psd_kernels.ExponentiatedQuadratic()

            with self.subTest(stddev=stddev, N=N):
                self.assert_tensors_almost_equal(
                    conditional_hsic([c, s], y, k), manual_chsic(c, s, y, k), delta=1e-3
                )

        with_grid_values({"stddev": [0.1, 0.5, 1.0, 5.0], "N": [5, 10, 50]}, once)

    def test_dhsic_same_input(self):
        def once(stddev, shape, n_vars):
            shape = (100, 10)
            stddev = 0.3
            y = np.random.choice([1, -1], shape)
            c = y + np.random.normal(0, stddev, y.shape)
            s = y + np.random.normal(0, stddev, y.shape)

            c = tf.cast(c, "float64")
            s = tf.cast(s, "float64")

            k = tfp.math.psd_kernels.ExponentiatedQuadratic()

            for n_vars in range(3, 11):
                many_vars = [c] * n_vars

                with self.subTest(n_vars=n_vars):
                    self.assert_tensors_almost_equal(
                        dhsic([c, c], k), dhsic(many_vars, k), delta=1e-3
                    )

            with_grid_values({**large_grid(), "n_vars": range(3, 11)}, once)

    def test_cka(self):
        def once(stddev, shape):
            y = np.random.choice([1, -1], shape)
            c = y + np.random.normal(0, stddev, y.shape)
            s = y + np.random.normal(0, stddev, y.shape)
            k = tfp.math.psd_kernels.ExponentiatedQuadratic()
            self.assert_tensors_almost_equal(cka([c, s], k), dcka([c, s], k))

        with_grid_values(large_grid(), once)

    def assert_tensors_almost_equal(self, t1, t2, delta=1e-5):
        return self.assertAlmostEqual(t1.numpy(), t2.numpy(), delta=delta)


if __name__ == "__main__":
    unittest.main()

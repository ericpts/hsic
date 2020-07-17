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
from typing import List, Tuple, Dict, Callable
import time


def center_matrix(M):
    M = M - tf.reduce_mean(M, axis=0)
    return M


def hsic(
    xs: List[tf.Tensor], k: tfp.math.psd_kernels.PositiveSemidefiniteKernel
) -> tf.Tensor:
    N = xs[0].shape[0]
    assert len(xs) == 2

    centered_gram_matrices = [center_matrix(k.matrix(f, f)) for f in xs]
    return tf.linalg.trace(centered_gram_matrices[0] @ centered_gram_matrices[1]) / (
        (N - 1) ** 2
    )


def unbiased_hsic(
    xs: List[tf.Tensor], k: tfp.math.psd_kernels.PositiveSemidefiniteKernel
) -> tf.Tensor:
    n = xs[0].shape[0]
    n_variables = len(xs)
    assert n_variables == 2

    matrices = [k.matrix(f, f) for f in xs]
    matrices = [m - tf.linalg.diag(tf.linalg.diag_part(m)) for m in matrices]

    tK = matrices[0]
    tL = matrices[0]

    score = (
        tf.linalg.trace(tK @ tL)
        + (tf.reduce_sum(tK) * tf.reduce_sum(tL) / (n - 1) / (n - 2))
        - (
            2
            * tf.tensordot(tf.reduce_sum(tK, axis=0), tf.reduce_sum(tL, axis=0), 0)
            / (n - 2)
        )
    )
    return score / n / (n - 3)


def cka(
    xs: List[tf.Tensor], k: tfp.math.psd_kernels.PositiveSemidefiniteKernel
) -> tf.Tensor:
    n_samples = xs[0].shape[0]
    n_variables = len(xs)

    H = tf.eye(n_samples) - 1 / n_samples
    centered_gram_matrices = [k.matrix(f, f) @ H for f in xs]

    pair_losses = []
    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            up = tf.linalg.trace(centered_gram_matrices[i] @ centered_gram_matrices[j])

            down_i = tf.linalg.trace(
                centered_gram_matrices[i] @ centered_gram_matrices[i]
            )

            down_j = tf.linalg.trace(
                centered_gram_matrices[j] @ centered_gram_matrices[j]
            )

            s = up / tf.sqrt(down_i * down_j)
            pair_losses.append(s)

    return tf.reduce_mean(pair_losses)


def label_kernel(l_0, l_1):
    l_0 = tf.one_hot(l_0, 1)
    l_1 = tf.one_hot(l_1, 1)
    return tfp.math.psd_kernels.ExponentiatedQuadratic().apply(l_0, l_1)


def conditional_hsic(
    xs: List[tf.Tensor],
    labels: tf.Tensor,
    k: tfp.math.psd_kernels.PositiveSemidefiniteKernel,
) -> tf.Tensor:
    n = xs[0].shape[0]
    n_variables = len(xs)
    H = tf.eye(n) - 1 / n

    def center(M):
        return M - tf.reduce_mean(M, axis=1)

    K = [k.matrix(f, f) for f in xs]
    one_hot_labels = tf.one_hot(labels, 1)
    K_z = k.matrix(one_hot_labels, one_hot_labels)

    K_xx = [center(x * K_z) for x in K]
    K = [center(x) for x in K]
    K_z = center(K_z)

    eps = 0.001
    R = eps * tf.linalg.inv(K_z + eps * tf.eye(n))

    pair_losses = []
    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            X = R @ K_xx[i] @ R
            Y = R @ K[j] @ R
            pair_losses.append(tf.linalg.trace(X @ Y))
    ret = tf.reduce_mean(pair_losses) / n

    return ret


def manual_chsic(c, s, y, k):
    unique_labels, _ = tf.unique(y)
    unique_labels = sorted(list(unique_labels.numpy()))

    n = c.shape[0]
    C = {z: [i for i in range(n) if y[i] == z] for z in unique_labels}
    s1 = 0.0
    for i in range(n):
        for j in range(n):
            s1 += k(c[i], c[j]) * k(s[i], s[j]) * label_kernel(y[i], y[j])
    s1 /= n

    s2 = 0.0
    for z in unique_labels:
        for zz in unique_labels:
            a1 = 0.0
            a2 = 0.0
            for i in C[z]:
                for ii in C[zz]:
                    a1 += k(c[i], c[ii])
                    a2 += k(s[i], s[ii])
            a = label_kernel(z, zz) * a1 * a2
            a /= len(C[z]) * len(C[zz])
            s2 += a
    s2 /= n

    s3 = 0.0
    for z in unique_labels:
        for i in range(n):
            t_0 = 0.0
            for ii in C[z]:
                t_0 += k(c[i], c[ii])
            t_1 = 0.0
            for jj in C[z]:
                t_1 += k(s[i], s[jj])
            t = t_0 * t_1
            t *= label_kernel(y[i], z)
            t /= len(C[z])
            s3 += t
    s3 /= n

    return s1 + s2 - 2 * s3


@gin.configurable
def diversity_loss(
    features: List[tf.Tensor], y: tf.Tensor, independence_measure: str, kernel: str,
) -> tf.Tensor:
    if kernel == "linear":
        k = tfp.math.psd_kernels.Linear()
    elif kernel == "rbf":
        k = tfp.math.psd_kernels.ExponentiatedQuadratic()
    else:
        raise ValueError(f"Unknown kernel: {kernel}; should be one of linear, rbf")

    if independence_measure == "cka":
        return cka(features, k)
    elif independence_measure == "hsic":
        return hsic(features, k)
    elif independence_measure == "unbiased_hsic":
        return unbiased_hsic(features, k)
    elif independence_measure == "conditional_hsic":
        return conditional_hsic(features, y, k)
    else:
        raise ValueError(
            f"Unknown independence_measure: {independence_measure}; expected one of cka or hsic"
        )


def label_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = tf.squeeze(y_true)
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    )


def forward(X: tf.Tensor, y_true: tf.Tensor, models: List[tf.keras.Model]) -> tf.Tensor:
    features = []
    ys_pred = []
    for m in models:
        (f, y_pred) = m(X)

        features.append(f)
        if y_pred.shape[-1] == 1:
            # In case of binary classification, having one logistic-regression
            # style logit is equivalent to having a second logit with fixed
            # value 0, and applying softmax to the both of them.
            y_pred = tf.concat((tf.zeros(y_pred.shape), y_pred), axis=-1)
        ys_pred.append(y_pred)

    return (features, ys_pred)


@gin.configurable
def compute_combined_loss(
    prediction_loss: tf.Tensor, div_loss: tf.Tensor, diversity_loss_coefficient: float,
) -> tf.Tensor:
    loss = 0
    loss += diversity_loss_coefficient * div_loss
    loss += tf.reduce_sum(prediction_loss)
    return loss


class Problem(object):
    def __init__(
        self,
        name: str,
        make_base_model: Callable[[], tf.keras.Model],
        batch_size: int = 256,
        n_models: int = 2,
    ) -> None:
        self.name = name
        self.batch_size = batch_size
        self.n_models = n_models

        self.models = [make_base_model() for i in range(self.n_models)]
        self.optimizer = tf.keras.optimizers.Nadam(lr=0.001, epsilon=0.001)

        self.variables = []
        for m in self.models:
            self.variables.extend(m.trainable_variables)

        self.init_metrics()
        self.init_logging()

    def reset_metrics(self):
        for ms in self.metrics:
            for m in ms:
                m.reset_states()

    def init_metrics(self):
        self.metrics = []
        for t in ["test", "train"]:
            ms = [
                tf.keras.metrics.Mean(f"{t}_prediction_loss", dtype=tf.float32)
                for i in range(self.n_models)
            ]
            self.metrics.append(ms)
            setattr(self, f"{t}_prediction_loss", ms)

            ms = [
                tf.keras.metrics.Accuracy(f"{t}_accuracy") for i in range(self.n_models)
            ]
            self.metrics.append(ms)
            setattr(self, f"{t}_accuracy", ms)

            m = tf.keras.metrics.Mean(f"{t}_diversity_loss", dtype=tf.float32)
            self.metrics.append([m])
            setattr(self, f"{t}_diversity_loss", m)

            m = tf.keras.metrics.Mean(f"{t}_combined_loss", dtype=tf.float32)
            self.metrics.append([m])
            setattr(self, f"{t}_combined_loss", m)

    def init_logging(self):
        self.base_log_dir = Path(
            f"logs/{self.name}/{datetime.now().strftime('%Y%m%d-%H%M%S')}/"
        )
        self.train_log_dir = self.base_log_dir / "train"
        self.train_summary_writer = tf.summary.create_file_writer(
            str(self.train_log_dir)
        )
        self.test_log_dir = self.base_log_dir / "test"
        self.test_summary_writer = tf.summary.create_file_writer(str(self.test_log_dir))

    @tf.function
    def train_step(self, X: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            (features, ys_pred) = forward(X, y, self.models)
            prediction_loss = [label_loss(y, y_pred) for y_pred in ys_pred]
            div_loss = diversity_loss(features, y)
            loss = compute_combined_loss(prediction_loss, div_loss)
            gradients = tape.gradient(loss, self.variables)

        self.optimizer.apply_gradients(zip(gradients, self.variables))

        for ip, p_loss in enumerate(prediction_loss):
            self.train_prediction_loss[ip](p_loss)
        self.train_diversity_loss(div_loss)

        for ip, y_pred in enumerate(ys_pred):
            self.train_accuracy[ip].update_state(
                y_true=y, y_pred=tf.math.argmax(y_pred, axis=1)
            )

        return loss

    @tf.function
    def test_step(self, X: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        (features, ys_pred) = forward(X, y, self.models)
        prediction_loss = [label_loss(y, y_pred) for y_pred in ys_pred]
        div_loss = diversity_loss(features, y)

        for ip, p_loss in enumerate(prediction_loss):
            self.test_prediction_loss[ip](p_loss)
        self.test_diversity_loss(div_loss)

        for ip, y_pred in enumerate(ys_pred):
            self.test_accuracy[ip].update_state(
                y_true=y, y_pred=tf.math.argmax(y_pred, axis=1)
            )

        return compute_combined_loss(prediction_loss, div_loss)

    def generate_training_data(self):
        raise NotImplemented("Must implement method in derived class!")

    def generate_testing_data(self):
        raise NotImplemented("Must implement method in derived class!")

    def train(self, epochs: int):
        D_train = self.generate_training_data()
        D_test = self.generate_testing_data()

        for epoch in range(epochs):
            t0 = time.time()
            self.reset_metrics()

            for (X, y) in D_test.batch(self.batch_size).prefetch(16):
                test_loss = self.test_step(X, y)
                self.test_combined_loss(test_loss)

            for (X, y) in D_train.batch(self.batch_size).prefetch(16):
                train_loss = self.train_step(X, y)
                self.train_combined_loss(train_loss)

            print(f"Epoch: {epoch + 1}")

            for ms in self.metrics:
                res = [m.result().numpy() for m in ms]
                metric_name = ms[0].name
                if len(res) == 1:
                    res = res[0]
                print(f"\t{metric_name}: {res}")
            print(f"\tTime per epoch: {time.time() - t0}")
            print("=" * 100)

        results = {}
        weights = {}
        for im, m in enumerate(self.models):
            cur_w = {}
            for w in m.trainable_variables:
                cur_w[w.name] = w.numpy().tolist()
            weights[im] = cur_w
        results["weights"] = weights

        for ms in self.metrics:
            res = [float(m.result().numpy()) for m in ms]
            metric_name = ms[0].name
            if len(res) == 1:
                res = res[0]
            results[metric_name] = res

        results["name"] = self.name
        results["diversity_loss_coefficient"] = gin.query_parameter(
            "compute_combined_loss.diversity_loss_coefficient"
        )
        return results

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


def hsic(
    xs: List[tf.Tensor], k: tfp.math.psd_kernels.PositiveSemidefiniteKernel
) -> tf.Tensor:
    n_samples = xs[0].shape[0]
    n_variables = len(xs)

    H = tf.eye(n_samples) - 1 / n_samples
    centered_gram_matrices = [k.matrix(f, f) @ H for f in xs]

    pair_losses = []
    for i in range(n_variables):
        for j in range(n_variables):
            if i == j:
                continue
            pair_losses.append(
                tf.linalg.trace(centered_gram_matrices[i] @ centered_gram_matrices[j])
            )
    return 1 / (n_samples - 1) ** 2 * tf.reduce_mean(pair_losses)


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


@gin.configurable
def diversity_loss(
    extracted_features: List[tf.Tensor], independence_measure: str, kernel: str,
) -> tf.Tensor:
    if kernel == "linear":
        k = tfp.math.psd_kernels.Linear()
    elif kernel == "rbf":
        k = tfp.math.psd_kernels.ExponentiatedQuadratic()
    else:
        raise ValueError(f"Unknown kernel: {kernel}; should be one of linear, rbf")

    if independence_measure == "cka":
        f_indep = cka
    elif independence_measure == "hsic":
        f_indep = hsic
    else:
        raise ValueError(
            f"Unknown independence_measure: {independence_measure}; expected one of cka or hsic"
        )
    return f_indep(extracted_features, k)


def label_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = tf.squeeze(y_true)
    return tf.keras.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred)


def forward(X: tf.Tensor, y_true: tf.Tensor, models: List[tf.keras.Model]) -> tf.Tensor:
    features = []
    ys_pred = []
    for m in models:
        (f, y_pred) = m(X)
        features.append(f)
        if y_pred.shape[-1] == 1:
            y_pred = tf.concat((tf.zeros(y_pred.shape), y_pred), axis=-1)
        probabilities = tf.nn.softmax(y_pred)
        ys_pred.append(probabilities)

    return (features, ys_pred)


@gin.configurable
def compute_combined_loss(
    prediction_loss: tf.Tensor,
    diversity_loss: tf.Tensor,
    diversity_loss_coefficient: float,
) -> tf.Tensor:
    return tf.reduce_mean(prediction_loss) + diversity_loss_coefficient * diversity_loss


class Problem(object):
    def __init__(
        self,
        name: str,
        make_base_model: Callable[[], tf.keras.Model],
        batch_size: int = 64,
        n_models: int = 2,
    ) -> None:
        self.name = name
        self.batch_size = batch_size
        self.n_models = n_models

        self.models = [make_base_model() for i in range(self.n_models)]
        self.optimizer = tf.keras.optimizers.Adam()
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
                tf.keras.metrics.Mean(f"{t}_accuracy", dtype=tf.float32)
                for i in range(self.n_models)
            ]
            self.metrics.append(ms)
            setattr(self, f"{t}_accuracy", ms)

            ms = [tf.keras.metrics.AUC(name=f"{t}_auroc") for i in range(self.n_models)]
            self.metrics.append(ms)
            setattr(self, f"{t}_auroc", ms)

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
            div_loss = diversity_loss(features)
            loss = compute_combined_loss(prediction_loss, div_loss)
            gradients = tape.gradient(loss, self.variables)

        for ip, p_loss in enumerate(prediction_loss):
            self.train_prediction_loss[ip](p_loss)
        self.train_diversity_loss(div_loss)

        for ip, y_pred in enumerate(ys_pred):
            self.train_accuracy[ip](
                tf.keras.metrics.sparse_categorical_accuracy(y_true=y, y_pred=y_pred)
            )
            self.train_auroc[ip].update_state(y_true=y, y_pred=y_pred[:, 1])

        self.optimizer.apply_gradients(zip(gradients, self.variables))
        return loss

    @tf.function
    def test_step(self, X: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        (features, ys_pred) = forward(X, y, self.models)
        prediction_loss = [label_loss(y, y_pred) for y_pred in ys_pred]
        div_loss = diversity_loss(features)
        for ip, p_loss in enumerate(prediction_loss):
            self.test_prediction_loss[ip](p_loss)
        self.test_diversity_loss(div_loss)

        for ip, y_pred in enumerate(ys_pred):
            self.test_accuracy[ip](
                tf.keras.metrics.sparse_categorical_accuracy(y_true=y, y_pred=y_pred)
            )
            self.test_auroc[ip].update_state(y_true=y, y_pred=y_pred[:, 1])

        return compute_combined_loss(prediction_loss, div_loss)

    def log_stats_to_tensorboard(self):
        with self.train_summary_writer.as_default():
            for ip, p_loss in enumerate(self.train_prediction_loss):
                tf.summary.scalar(f"prediction loss {ip}", p_loss.result(), step=epoch)

            for ip, acc in enumerate(self.train_accuracy):
                tf.summary.scalar(f"accuracy {ip}", acc.result(), step=epoch)

            for ip, auroc in enumerate(self.train_auroc):
                tf.summary.scalar(f"auroc {ip}", auroc.result(), step=epoch)

            tf.summary.scalar(
                "diversity loss", self.train_diversity_loss.result(), step=epoch
            )
            tf.summary.scalar(
                "combined loss", self.train_combined_loss.result(), step=epoch
            )

        with test_summary_writer.as_default():
            for ip, p_loss in enumerate(self.test_prediction_loss):
                tf.summary.scalar(f"prediction loss {ip}", p_loss.result(), step=epoch)

            for ip, acc in enumerate(self.test_accuracy):
                tf.summary.scalar(f"accuracy {ip}", acc.result(), step=epoch)

            for ip, auroc in enumerate(self.test_auroc):
                tf.summary.scalar(f"auroc {ip}", auroc.result(), step=epoch)

            tf.summary.scalar(
                "diversity loss", self.test_diversity_loss.result(), step=epoch
            )
            tf.summary.scalar(
                "combined loss", self.test_combined_loss.result(), step=epoch
            )

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
            for (X, y) in D_train.batch(self.batch_size).prefetch(16):
                train_loss = self.train_step(X, y)
                self.train_combined_loss(train_loss)

            for (X, y) in D_test.batch(16 * self.batch_size).prefetch(8):
                test_loss = self.test_step(X, y)
                self.test_combined_loss(test_loss)

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


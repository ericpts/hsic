import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import gin
import gin.tf
import argparse
from datetime import datetime
from pathlib import Path


from typing import List, Tuple


# Hyperparameters
batch_size = 64
N_MODELS = 2

train_prediction_loss = [
    tf.keras.metrics.Mean("train_prediction_loss_{i}", dtype=tf.float32)
    for i in range(N_MODELS)
]

train_diversity_loss = tf.keras.metrics.Mean("train_diversity_loss", dtype=tf.float32)
train_combined_loss = tf.keras.metrics.Mean("train_combined_loss", dtype=tf.float32)

test_prediction_loss = [
    tf.keras.metrics.Mean(f"test_prediction_loss_{i}", dtype=tf.float32)
    for i in range(N_MODELS)
]
test_diversity_loss = tf.keras.metrics.Mean("test_diversity_loss", dtype=tf.float32)
test_combined_loss = tf.keras.metrics.Mean("test_combined_loss", dtype=tf.float32)


def base_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(2)
    X = inputs
    X = tf.keras.layers.Dense(1, activation="linear")(X)
    feature_extractor = X
    X = tf.keras.layers.Dense(1, activation="linear")(X)

    return tf.keras.Model(inputs, outputs=[feature_extractor, X])


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


def cov_hsic(f, g):
    n_samples = f.shape[0]
    cov = 1 / (n_samples - 1) * (f.T @ g)
    score = np.linalg.norm(cov)
    return score ** 2.0


def cov_cka(f, g):
    up = cov_hsic(f, g)
    down_f = cov_hsic(f, f)
    down_g = cov_hsic(g, g)
    s = up / np.sqrt(down_f * down_g)

    return up / np.sqrt(down_f * down_g)


@gin.configurable
def diversity_loss(
    extracted_features: List[tf.Tensor], kernel: str = "linear",
) -> tf.Tensor:
    if kernel == "linear":
        k = tfp.math.psd_kernels.Linear()
    elif kernel == "rbf":
        k = tfp.math.psd_kernels.ExponentiatedQuadratic()
    else:
        raise ValueError(f"Unknown kernel: {kernel}; should be one of linear, rbf")
    return cka(extracted_features, k)


def label_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_pred = tf.concat((tf.zeros(y_pred.shape), y_pred), axis=-1)
    y_true = tf.reshape(tf.where(y_true > 0, x=1, y=0), (-1,))
    return tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)
    # return tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))


def cross_covariance(x: np.array, y: np.array) -> float:
    cov = 1 / (x.shape[0] - 1) * np.linalg.norm(x.T @ y)
    cov = cov ** 2
    return cov


def forward(X: tf.Tensor, y_true: tf.Tensor, ms: List[tf.keras.Model]) -> tf.Tensor:
    features = []
    ys_pred = []
    for m in ms:
        (f, y_p) = m(X)
        features.append(f)
        ys_pred.append(y_p)

    prediction_loss = [label_loss(y_true, y_pred) for y_pred in ys_pred]
    div_loss = diversity_loss(features)
    return (prediction_loss, div_loss)


def generate_training_data(
    n_samples: int = 100_000, sigma_c: float = 1.0, sigma_s: float = 1.0
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    y = np.random.choice([1, -1], size=(n_samples, 1))

    c = np.random.normal(y, scale=sigma_c, size=(n_samples, 1))
    s = np.random.normal(y, scale=sigma_s, size=(n_samples, 1))

    X = np.concatenate([c, s], axis=-1)

    return tf.data.Dataset.from_tensor_slices((X, tf.cast(y, tf.float32)))


def generate_testing_data(
    n_samples: int = 100_000, sigma_c: float = 1.0, sigma_s: float = 1.0
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    y = np.random.choice([1, -1], size=(n_samples, 1))
    c = np.random.normal(y, scale=sigma_c, size=(n_samples, 1))

    z = np.random.choice([1, -1], size=(n_samples, 1))
    s = np.random.normal(z, scale=sigma_s, size=(n_samples, 1))

    X = np.concatenate([c, s], axis=-1)

    return tf.data.Dataset.from_tensor_slices((X, tf.cast(y, tf.float32)))


ms = [base_model() for i in range(N_MODELS)]
optimizer = tf.keras.optimizers.Adam()
variables = []
for m in ms:
    variables.extend(m.trainable_variables)


@gin.configurable
def compute_combined_loss(
    prediction_loss: tf.Tensor,
    diversity_loss: tf.Tensor,
    diversity_loss_coefficient: float = 1 / 1,
) -> tf.Tensor:
    return tf.reduce_mean(prediction_loss) + diversity_loss_coefficient * diversity_loss


@tf.function
def train_step(X: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    with tf.GradientTape() as tape:
        prediction_loss, div_loss = forward(X, y, ms)
        for ip, p_loss in enumerate(prediction_loss):
            train_prediction_loss[ip](p_loss)
        train_diversity_loss(div_loss)

        loss = compute_combined_loss(prediction_loss, div_loss)
        gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))
    return loss


@tf.function
def test_step(X: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    prediction_loss, div_loss = forward(X, y, ms)
    for ip, p_loss in enumerate(prediction_loss):
        test_prediction_loss[ip](p_loss)
    test_diversity_loss(div_loss)
    return compute_combined_loss(prediction_loss, div_loss)


@gin.configurable
def main_loop(epochs: int = 30):
    D_train = generate_training_data().shuffle(100_000).batch(batch_size)
    D_test = generate_testing_data(n_samples=10_000).batch(batch_size * 16)

    # Logging setup
    base_log_dir = Path(f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}/")
    train_log_dir = base_log_dir / "train"
    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
    test_log_dir = base_log_dir / "test"
    test_summary_writer = tf.summary.create_file_writer(str(test_log_dir))

    for epoch in range(30):
        for (X, y) in D_train.prefetch(8):
            train_loss = train_step(X, y)
            train_combined_loss(train_loss)

        with train_summary_writer.as_default():
            for ip, p_loss in enumerate(train_prediction_loss):
                tf.summary.scalar(f"prediction loss {ip}", p_loss.result(), step=epoch)
            tf.summary.scalar(
                "diversity loss", train_diversity_loss.result(), step=epoch
            )
            tf.summary.scalar("combined loss", train_combined_loss.result(), step=epoch)

            if False:
                for im, m in enumerate(ms):
                    vs = m.trainable_variables
                    feature_extractor_weights = vs[0]
                    print(f"weights for network {im}: {vs}")
                    tf.summary.histogram(
                        f"weights {im}", feature_extractor_weights, step=epoch
                    )

        for (X, y) in D_test.prefetch(8):
            test_loss = test_step(X, y)
            test_combined_loss(test_loss)

        with test_summary_writer.as_default():
            for ip, p_loss in enumerate(test_prediction_loss):
                tf.summary.scalar(f"prediction loss {ip}", p_loss.result(), step=epoch)
            tf.summary.scalar(
                "diversity loss", test_diversity_loss.result(), step=epoch
            )
            tf.summary.scalar("combined loss", test_combined_loss.result(), step=epoch)

        print(
            f"Epoch: {epoch + 1}; \
            train prediction: {[t.result().numpy() for t in train_prediction_loss]}; train diversity:\
            {train_diversity_loss.result()};\
            test prediction: {[t.result().numpy() for t in test_prediction_loss]}; test diversity:\
            {test_diversity_loss.result()};"
        )

        for t in train_prediction_loss:
            t.reset_states()
        train_diversity_loss.reset_states()
        train_combined_loss.reset_states()

        for t in test_prediction_loss:
            t.reset_states()
        test_diversity_loss.reset_states()
        test_combined_loss.reset_states()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gin_config_file", type=str)
    args = parser.parse_args()
    if args.gin_config_file:
        gin.parse_config_file(args.gin_config_file)

    main_loop()


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import gin
import gin.tf
import argparse
from datetime import datetime
from pathlib import Path
import colour_mnist


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
    inputs = tf.keras.layers.Input((28, 28, 3))
    X = inputs
    kernel_size = 7
    X = tf.keras.layers.Conv2D(16, 7, padding="SAME", activation="relu")(X)
    X = tf.keras.layers.Conv2D(32, 7, padding="SAME", activation="relu")(X)
    #     X = tf.keras.layers.Conv2D(64, 7, padding="SAME", activation="relu")(X)
    #     X = tf.keras.layers.Conv2D(128, 7, padding="SAME", activation="relu")(X)
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(10, activation="relu")(X)
    feature_extractor = X
    X = tf.keras.layers.Dense(10, activation="linear")(X)

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
    return tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)
    # return tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))


def forward(X: tf.Tensor, y_true: tf.Tensor, ms: List[tf.keras.Model]) -> tf.Tensor:
    features = []
    ys_pred = []
    for m in ms:
        (f, y_p) = m(X)
        f = tf.debugging.assert_all_finite(f, f"expected features to be finite")
        y_p = tf.debugging.assert_all_finite(y_p, f"expected predictions to be finite")
        features.append(f)
        ys_pred.append(y_p)

    prediction_loss = [label_loss(y_true, y_pred) for y_pred in ys_pred]
    div_loss = diversity_loss(features)
    return (prediction_loss, div_loss)


def generate_training_data() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    X, y, biased_y = colour_mnist.biased_mnist(
        "~/.datasets/mnist/", batch_size, 0.8, train=True
    )

    return tf.data.Dataset.from_tensor_slices((X, (y, biased_y)))


def generate_testing_data() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    X, y, biased_y = colour_mnist.biased_mnist(
        "~/.datasets/mnist/", batch_size, 0.1, train=False
    )
    return tf.data.Dataset.from_tensor_slices((X, (y, biased_y)))


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

    # Logging setup
    base_log_dir = Path(f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}/")
    train_log_dir = base_log_dir / "train"
    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
    test_log_dir = base_log_dir / "test"
    test_summary_writer = tf.summary.create_file_writer(str(test_log_dir))

    D_train = generate_training_data().shuffle(100_000).batch(batch_size)
    D_test = generate_testing_data().batch(batch_size * 16)

    for epoch in range(epochs):
        for (X, (y, y_biased)) in D_train.prefetch(8):
            train_loss = train_step(X, y)
            train_combined_loss(train_loss)
            print(".", end="", flush=True)
        print("")

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

        for (X, (y, y_biased)) in D_test.prefetch(8):
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

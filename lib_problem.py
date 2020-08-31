import tensorflow as tf
import gin
import gin.tf
from datetime import datetime
from pathlib import Path
from typing import Callable, List
import time
from lib_independence_metrics import diversity_loss
import lib_mlflow
import lib_auroc


@gin.configurable
def get_weight_regularizer(strength: float = 0.01):
    return tf.keras.regularizers.l2(strength)


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
    loss = tf.reduce_sum(prediction_loss)
    if div_loss != 0.0:
        loss += diversity_loss_coefficient * div_loss
    return loss


@gin.configurable
class Problem(object):
    def __init__(
        self,
        name: str,
        make_base_model: Callable[[], tf.keras.Model],
        batch_size: int = 256,
        n_models: int = 2,
        initial_lr: float = 0.001,
        n_epochs: int = 100,
        decrease_lr_at_epochs: List[int] = [20, 40, 80],
        optimizer=tf.keras.optimizers.Nadam,
    ) -> None:
        self.name = name
        self.batch_size = batch_size
        self.n_models = n_models

        self.models = [make_base_model() for i in range(self.n_models)]
        self.lr = tf.Variable(initial_lr)

        self.optimizer = optimizer(lr=self.lr)

        self.decrease_lr_at_epochs = decrease_lr_at_epochs
        self.n_epochs = n_epochs

        self.variables = []
        for m in self.models:
            self.variables.extend(m.trainable_variables)

        self.init_metrics()
        self.init_logging()

    def reset_metrics(self):
        for ms in self.metrics.values():
            for m in ms:
                m.reset_states()

    def init_metrics(self):

        Mean = tf.keras.metrics.Mean
        Accuracy = tf.keras.metrics.Accuracy
        self.metrics = {}
        for t in ["test_id", "test_ood", "train"]:

            def add_metric(
                metric_f, base_name: str, is_per_model: bool = True, **kwargs
            ):
                if is_per_model:
                    n_copies = self.n_models
                else:
                    n_copies = 1
                ms = [metric_f(f"{t}_{base_name}", **kwargs) for i in range(n_copies)]
                self.metrics[f"{t}_{base_name}"] = ms

            add_metric(Mean, "prediction_loss", is_per_model=True)
            add_metric(Accuracy, "accuracy", is_per_model=True)

            add_metric(Mean, "diversity_loss", is_per_model=False)
            add_metric(Mean, "combined_loss", is_per_model=False)
            add_metric(Accuracy, "ensemble_accuracy", is_per_model=False)

        def add_auroc_metric(metric_name: str):
            metric_name = f"auroc_{metric_name}"
            m = tf.keras.metrics.Sum(metric_name)
            self.metrics[metric_name] = [m]

        for m in ["pmax", "entropy", "max_diff", "average_diff"]:
            add_auroc_metric(m)

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
    def forward_step(self, X: tf.Tensor, y: tf.Tensor, metric_prefix: str) -> tf.Tensor:
        (features, ys_pred) = forward(X, y, self.models)
        prediction_loss = [label_loss(y, y_pred) for y_pred in ys_pred]
        div_loss = diversity_loss(features, y)
        loss = compute_combined_loss(prediction_loss, div_loss)

        for ip, p_loss in enumerate(prediction_loss):
            self.metrics[f"{metric_prefix}_prediction_loss"][ip](p_loss)
        self.metrics[f"{metric_prefix}_diversity_loss"][0](div_loss)

        probabilities = [tf.nn.softmax(y_pred, axis=-1) for y_pred in ys_pred]
        ensemble_probabilities = tf.math.reduce_mean(probabilities, axis=0)
        self.metrics[f"{metric_prefix}_ensemble_accuracy"][0].update_state(
            y_true=y, y_pred=tf.math.argmax(ensemble_probabilities, axis=1)
        )

        for ip, y_pred in enumerate(ys_pred):
            self.metrics[f"{metric_prefix}_accuracy"][ip].update_state(
                y_true=y, y_pred=tf.math.argmax(y_pred, axis=1)
            )

        self.metrics[f"{metric_prefix}_combined_loss"][0](loss)

        return loss, features, ys_pred

    @tf.function
    def train_step(self, X: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss, features, ys_pred = self.forward_step(X, y, "train")
            gradients = tape.gradient(loss, self.variables)

        self.optimizer.apply_gradients(zip(gradients, self.variables))
        return loss, features, ys_pred

    @tf.function
    def test_step(self, X: tf.Tensor, y: tf.Tensor, distribution: str) -> tf.Tensor:
        assert distribution in ["ood", "id"]
        metric_prefix = f"test_{distribution}"
        return self.forward_step(X, y, metric_prefix)

    def generate_training_data(self):
        raise NotImplemented("Must implement method in derived class!")

    def generate_id_testing_data(self):
        raise NotImplemented("Must implement method in derived class!")

    def generate_ood_testing_data(self):
        raise NotImplemented("Must implement method in derived class!")

    def on_epoch_start(self, epoch: int):
        self.reset_metrics()
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch: int):
        def log_metrics_to_tensorboard():
            with self.train_summary_writer.as_default():
                for ms in self.metrics.values():
                    for m in ms:
                        if not m.name.startswith("train"):
                            continue
                        tf.summary.scalar(m.name, m.result(), step=epoch)

            with self.test_summary_writer.as_default():
                for ms in self.metrics.values():
                    for m in ms:
                        if not m.name.startswith("test"):
                            continue
                        tf.summary.scalar(m.name, m.result(), step=epoch)

        def log_metrics_to_console():
            for ms in self.metrics.values():
                res = [m.result().numpy() for m in ms]
                metric_name = ms[0].name
                if len(res) == 1:
                    res = res[0]
                print(f"\t{metric_name}: {res}")

        def log_metrics_to_mlflow():
            for ms in self.metrics.values():
                if len(ms) == 1:
                    m = ms[0]
                    lib_mlflow.log_metric(m.name, m.result().numpy(), step=epoch)
                    continue

                for im, m in enumerate(ms):
                    lib_mlflow.log_metric(
                        f"{m.name}_{im}", m.result().numpy(), step=epoch
                    )

        log_metrics_to_tensorboard()
        log_metrics_to_mlflow()
        print(f"Epoch: {epoch + 1}")
        log_metrics_to_console()
        print(f"Took {time.time() - self.epoch_start_time:.2f} seconds.")
        print("=" * 100)

        if (
            len(self.decrease_lr_at_epochs) > 0
            and epoch == self.decrease_lr_at_epochs[0]
        ):
            self.lr.assign(self.lr.value() / 10.0)
            self.decrease_lr_at_epochs.pop(0)

    def train(self):
        D_train = self.generate_training_data()
        D_test_id = self.generate_id_testing_data()
        D_test_ood = self.generate_ood_testing_data()

        n_train = 0
        for X, y in D_train:
            n_train += 1

        n_test_id = 0
        for X, y in D_test_id:
            n_test_id += 1

        n_test_ood = 0
        for X, y in D_test_ood:
            n_test_ood += 1

        print(
            f"Samples: Train: {n_train}; Test id: {n_test_id}; Test ood: {n_test_ood}."
        )

        D_train = (
            D_train.shuffle(200_000)
            .batch(self.batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        D_test_id = D_test_id.batch(self.batch_size).prefetch(
            tf.data.experimental.AUTOTUNE
        )
        D_test_ood = D_test_ood.batch(self.batch_size).prefetch(
            tf.data.experimental.AUTOTUNE
        )

        print("Starting training...")
        for epoch in range(self.n_epochs):
            self.on_epoch_start(epoch)

            for (X, y) in D_train:
                self.train_step(X, y)

            id_y_hat = []
            for (X, y) in D_test_id:
                loss, features, ys_pred = self.test_step(X, y, "id")
                id_y_hat.append(tf.transpose(tf.nn.softmax(ys_pred), [1, 0, 2]))

            ood_y_hat = []
            for (X, y) in D_test_ood:
                loss, features, ys_pred = self.test_step(X, y, "ood")
                ood_y_hat.append(tf.transpose(tf.nn.softmax(ys_pred), [1, 0, 2]))

            id_y_hat = tf.concat(id_y_hat, axis=0)
            ood_y_hat = tf.concat(ood_y_hat, axis=0)

            for (k, v) in lib_auroc.compute_metrics(id_y_hat, ood_y_hat).items():
                self.metrics[k][0].update_state(v)

            self.on_epoch_end(epoch)

        results = {}
        for ms in self.metrics.values():
            res = [float(m.result().numpy()) for m in ms]
            metric_name = ms[0].name
            if len(res) == 1:
                res = res[0]
            results[metric_name] = res
        results["name"] = self.name
        return results, self.models

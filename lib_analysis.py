from collections import defaultdict
from pathlib import Path
from plotly.colors import n_colors
from plotly.subplots import make_subplots
from typing import Dict, Any, Callable
from typing import List, Optional
import argparse
import concurrent.futures
import dash_core_components as dcc
import dash_html_components as html
import gin
import json
import json
import lib_problem
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import shutil
import subprocess
import tensorflow as tf
import tensorflow_probability as tfp
import yaml

_gin_columns_rename = {
    "compute_combined_loss.diversity_loss_coefficient": "lambda",
    "diversity_loss.kernel": "kernel",
    "diversity_loss.independence_measure": "indep",
    "BiasedMnistProblem.training_data_label_correlation": "label_correlation",
}


def _parse_gin_config(config: str) -> Dict[str, Any]:
    lines = config.split("\n")
    ret = {}
    for line in lines:
        tokens = [t.strip() for t in line.split("=")]
        tokens = [t for t in tokens if t]
        if len(tokens) == 0:
            continue
        assert len(tokens) == 2, f"Got unexpected line: {tokens}"
        key, value = tokens
        ret[key] = eval(value)
    return ret


def read_problem(root: Path, problem: str) -> pd.DataFrame:
    rows = []
    for yaml_config in (root / problem).glob("**/config.yaml"):
        with open(yaml_config, "rt") as f:
            cfg_dict = yaml.load(f, Loader=yaml.CLoader)

        results_json = root / cfg_dict["results_json_output"]
        if not results_json.exists():
            print(f"Could not find {results_json}! Skipping folder...")
            continue

        with results_json.open("rt") as f:
            results_dict = json.load(f)

        gin_config = _parse_gin_config((root / cfg_dict["gin_config_file"]).read_text())
        row = {**results_dict, **gin_config}
        row["original_config"] = yaml_config

        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.rename(columns=_gin_columns_rename)
    return df


def l2_probability_distance(y_hat: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum((y_hat[:, 0, :] - y_hat[:, 1, :]) ** 2.0, axis=-1)


def l1_probability_distance(y_hat: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(tf.abs(y_hat[:, 0, :] - y_hat[:, 1, :]), axis=-1)


def plot_digit_grid(X: tf.Tensor, y: tf.Tensor, per_digit: int = 10):
    f, axarr = plt.subplots(per_digit, 10)

    for i in range(10):
        x = X[y == i]
        if x.shape[0] == 0:
            print(f"Could not find any images with label {i}! Skipping...")
            continue
        indices = np.random.choice(x.shape[0], size=per_digit)
        for j in range(per_digit):
            axarr[j, i].imshow(x[indices[j]] * 0.5 + 0.5)
            axarr[j, i].axis("off")

    plt.show()


def plot_biased_digit_grid(X: tf.Tensor, y: tf.Tensor, y_biased: tf.Tensor):
    n_digits = len(np.unique(y))

    f, axarr = plt.subplots(10, n_digits)

    for i in range(n_digits):
        for j in range(10):
            x = X[(y == i) & (y_biased == j)]
            if x.shape[0] == 0:
                print(
                    f"Could not find any images with label {i} and bias {j}! Skipping..."
                )
                continue
            indices = np.random.choice(x.shape[0], size=1)
            axarr[j, i].imshow(x[indices[0]] * 0.5 + 0.5)
            axarr[j, i].axis("off")

    plt.show()


def pretty_print_matrix(matrix: np.ndarray) -> go.Figure:
    colors = n_colors("rgb(200, 200, 250)", "rgb(250, 200, 200)", 101, colortype="rgb")
    colors = np.array(colors)

    bgs = matrix.shape[1]

    header_values = ["image digit"] + [f"bg-color-{i}" for i in range(bgs)]

    n_rows = matrix.shape[0]
    n_cols = matrix.shape[1]

    values_of_column = np.ndarray((n_cols, n_rows), dtype=object)
    percentages = np.ndarray((n_cols, n_rows), dtype=float)

    for i in range(n_rows):
        for j in range(n_cols):
            (correct, total) = matrix[i, j, :]
            if total > 0:
                percent = correct / total * 100
            else:
                percent = 0

            percentages[j][i] = percent
            values_of_column[j][i] = f"{correct} / {total} ({percent:.2f}%)"

    percentages = np.around(percentages).astype(np.int32)

    fill_color = []

    for col in percentages:
        add = colors[col]
        fill_color.append(add)

    row_legend = np.asarray([str(i) for i in range(n_rows)])[np.newaxis, :]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header_values,
                    line_color="darkslategray",
                    fill_color="lightskyblue",
                    align="left",
                ),
                cells=dict(
                    values=np.concatenate([row_legend, values_of_column], axis=0),
                    line_color="darkslategray",
                    fill_color=[["lightskyblue"] * n_rows] + fill_color,
                    align="left",
                ),
            )
        ]
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=0, pad=0), paper_bgcolor="LightSteelBlue",
    )
    fig.update_layout(width=1000, height=150)
    fig.update_yaxes(automargin=True)
    return fig


def make_confusion_matrix(
    X: tf.Tensor, y: tf.Tensor, y_biased: tf.Tensor, y_label_hat: tf.Tensor
):
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == y_biased.shape[0]
    assert X.shape[0] == y_label_hat.shape[0]

    digits = np.unique(y)
    backgrounds = list(range(10))

    digit_accuracy_matrix = np.ndarray(
        ((np.max(digits) + 1), len(backgrounds)), dtype=(int, 2)
    )

    for i in digits:
        for j in backgrounds:
            select = (y == i) & (y_biased == j)
            total_entries = tf.math.count_nonzero(select)
            accuracy = tf.math.count_nonzero(y_label_hat[select] == y[select])
            bias = tf.math.count_nonzero(y_label_hat[select] == y_biased[select])
            digit_accuracy_matrix[i, j] = (accuracy, total_entries)
    return digit_accuracy_matrix


def _compute_overall_statistics(
    X: tf.Tensor, y: tf.Tensor, y_biased: tf.Tensor, y_hat: tf.Tensor,
):
    n_total = y.shape[0]

    ensemble_y_label_hat = tf.argmax(tf.math.reduce_mean(y_hat, 1), axis=-1)
    ensemble_accuracy = tf.math.count_nonzero(ensemble_y_label_hat == y) / n_total

    y_label_hat = tf.argmax(y_hat, axis=-1)
    model_accuracy = [
        tf.math.count_nonzero(y_label_hat[:, im] == y) / n_total for im in range(2)
    ]

    return html.Div(
        [
            html.H5("Overall statistics"),
            html.Div(
                [
                    html.H6("Ensemble"),
                    html.P(f"Accuracy: {ensemble_accuracy * 100:.2f}%"),
                    dcc.Graph(
                        figure=pretty_print_matrix(
                            make_confusion_matrix(X, y, y_biased, ensemble_y_label_hat)
                        )
                    ),
                ]
            ),
            *[
                html.Div(
                    [
                        html.H6(f"Model {im}"),
                        html.P(f"Accuracy: {model_accuracy[im] * 100:.2f}%"),
                        dcc.Graph(
                            figure=pretty_print_matrix(
                                make_confusion_matrix(
                                    X, y, y_biased, tf.argmax(y_hat[:, im], axis=-1)
                                )
                            )
                        ),
                    ]
                )
                for im in range(2)
            ],
        ]
    )


def _compute_disagreement_statistics(
    X: tf.Tensor, y: tf.Tensor, y_biased: tf.Tensor, y_hat: tf.Tensor,
):
    y_label_hat = tf.argmax(y_hat, axis=-1)

    n_original = y.shape[0]
    select = y_label_hat[:, 0] != y_label_hat[:, 1]
    n_select = tf.math.count_nonzero(select)

    if n_select == 0:
        return html.H5("No disagreement")

    X = X[select]
    y = y[select]
    y_biased = y_biased[select]
    y_label_hat = y_label_hat[select]

    model_accuracy = [
        tf.math.count_nonzero(y_label_hat[:, im] == y) / n_select for im in range(2)
    ]

    return html.Div(
        [
            html.H5("On disagreement"),
            html.P(
                f"Total % of data: {n_select} / {n_original} ({n_select / n_original * 100:.2f}%)"
            ),
            *[
                html.Div(
                    [
                        html.H6(f"Model {im}"),
                        html.P(f"Accuracy: {model_accuracy[im] * 100:.2f}%"),
                        dcc.Graph(
                            figure=pretty_print_matrix(
                                make_confusion_matrix(
                                    X, y, y_biased, y_label_hat[:, im]
                                )
                            )
                        ),
                    ]
                )
                for im in range(2)
            ],
        ]
    )


def _compute_agreement_statistics(
    X: tf.Tensor, y: tf.Tensor, y_biased: tf.Tensor, y_hat: tf.Tensor,
):
    y_label_hat = tf.argmax(y_hat, axis=-1)

    n_original = y.shape[0]
    select = y_label_hat[:, 0] == y_label_hat[:, 1]
    n_select = tf.math.count_nonzero(select)

    X = X[select]
    y = y[select]
    y_biased = y_biased[select]
    y_label_hat = y_label_hat[select][:, 0]

    acc = tf.math.count_nonzero(y_label_hat == y) / n_select
    return html.Div(
        [
            html.H5("On agreement"),
            html.P(
                f"Total % of data: {n_select} / {n_original} ({n_select / n_original * 100:.2f}%)"
            ),
            html.P(f"Accuracy: {acc * 100:.2f}%"),
            html.Div(
                [
                    html.H6(f"Overall confusion matrix"),
                    dcc.Graph(
                        figure=pretty_print_matrix(
                            make_confusion_matrix(X, y, y_biased, y_label_hat)
                        )
                    ),
                ]
            ),
        ]
    )


def print_statistics(
    X: tf.Tensor, y: tf.Tensor, y_biased: tf.Tensor, y_hat: tf.Tensor,
):
    overall_statistics = _compute_overall_statistics(X, y, y_biased, y_hat)
    disagreement_statistics = _compute_disagreement_statistics(X, y, y_biased, y_hat)
    agreement_statistics = _compute_agreement_statistics(X, y, y_biased, y_hat)

    return [overall_statistics, disagreement_statistics, agreement_statistics]


def process_dataset(
    dataset: tf.data.Dataset,
    models: List[tf.keras.Model],
    batch_size: int,
    include_div_losses: bool = False,
    kernel: str = "unbiased_hsic",
) -> tf.Tensor:
    Xs = []
    true_labels = []
    predicted = []
    biased_labels = []
    div_losses = []
    for X, y, y_biased in dataset.batch(batch_size):
        Xs.append(X)
        (features, ys_pred) = lib_problem.forward(X, y, models)

        probabilities = tf.nn.softmax(ys_pred)
        probabilities = tf.transpose(probabilities, [1, 0, 2])
        true_labels.append(y)
        biased_labels.append(y_biased)
        predicted.append(probabilities)

        div = lib_problem.diversity_loss(features, y, kernel, "rbf").numpy()
        div_losses.append(div)

    Xs = tf.concat(Xs, 0)
    true_labels = tf.concat(true_labels, 0)
    biased_labels = tf.concat(biased_labels, 0)
    predicted = tf.concat(predicted, 0)
    div_losses = tf.concat(div_losses, 0)

    true_labels = tf.cast(true_labels, tf.int64)
    biased_labels = tf.cast(biased_labels, tf.int64)

    if include_div_losses:
        return Xs, true_labels, biased_labels, predicted, div_losses
    else:
        return Xs, true_labels, biased_labels, predicted


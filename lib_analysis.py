from collections import defaultdict
from pathlib import Path
from plotly.colors import n_colors
from typing import Dict, Any, Callable
from typing import List, Optional
import argparse
import concurrent.futures
import gin
import json
import json
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


def _read_problem_raw_data(root: Path, problem: str) -> pd.DataFrame:
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


def _weights_to_numpy(weights: pd.Series):
    ret = {}
    for inet in weights.keys():
        ret[inet] = {}
        for ilayer, wlayer in weights[inet].items():
            ret[inet][ilayer] = np.asarray(wlayer)
    return ret


def _process_weights_for_cos_and_norm(weights: pd.Series):
    w0 = weights["0"]["dense/kernel:0"]
    w1 = weights["1"]["dense_2/kernel:0"]

    norm_0 = float(np.linalg.norm(w0))
    norm_1 = float(np.linalg.norm(w1))

    w0_normed = w0.T / np.linalg.norm(w0, axis=0)[:, np.newaxis]
    w1_normed = w1.T / np.linalg.norm(w1, axis=0)[:, np.newaxis]

    c = w0_normed @ w1_normed.T
    c = np.absolute(c)
    assert (c <= 1).all()
    c = c.max()
    ret = pd.Series({"cos": c, "norm": (norm_0, norm_1)})
    return ret


def read_problem(root: Path, problem: str) -> pd.DataFrame:
    DF = _read_problem_raw_data(root, problem)
    DF["weights"] = DF["weights"].apply(_weights_to_numpy)
    DF[["cos", "norm"]] = DF["weights"].apply(_process_weights_for_cos_and_norm)
    return DF


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


def pretty_print_matrix(matrix: np.ndarray):
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
    fig.show()


def for_one_confusion_matrix(
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

    bias_matrix = np.ndarray(((np.max(digits) + 1), len(backgrounds)), dtype=(int, 2))

    for i in digits:
        for j in backgrounds:
            select = (y == i) & (y_biased == j)
            total_entries = tf.math.count_nonzero(select)
            accuracy = tf.math.count_nonzero(y_label_hat[select] == y[select])
            bias = tf.math.count_nonzero(y_label_hat[select] == y_biased[select])
            digit_accuracy_matrix[i, j] = (accuracy, total_entries)
            bias_matrix[i, j] = (bias, total_entries)
    return digit_accuracy_matrix, bias_matrix


def render_confusion_matrices(
    X: tf.Tensor,
    y: tf.Tensor,
    y_biased: tf.Tensor,
    y_label_hat: tf.Tensor,
    models: List[int] = [0, 1],
    differences: bool = False,
):
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == y_biased.shape[0]
    assert X.shape[0] == y_label_hat.shape[0]
    models_confusion_matrix = [
        for_one_confusion_matrix(X, y, y_biased, y_label_hat[:, im]) for im in [0, 1]
    ]

    for im in models:
        print(f"Model {im}:")
        print(f"\tPredicted digit:")
        pretty_print_matrix(models_confusion_matrix[im][0])
        print(f"\tPredicted bg:")
        pretty_print_matrix(models_confusion_matrix[im][1])

    if differences:
        print("Differences: ")
        print("\tDigit:")
        pretty_print_df(
            pd.DataFrame(
                np.absolute(
                    models_confusion_matrix[0][0] - models_confusion_matrix[1][0]
                )
            )
        )
        print("\tBg:")
        pretty_print_df(
            pd.DataFrame(
                np.absolute(
                    models_confusion_matrix[0][1] - models_confusion_matrix[1][1]
                )
            )
        )


def print_statistics(
    X: tf.Tensor,
    y: tf.Tensor,
    y_biased: tf.Tensor,
    y_hat: tf.Tensor,
    print_digits: bool = False,
    print_confusion_matrices: bool = False,
):
    y_label_hat = tf.argmax(y_hat, axis=-1, output_type=tf.int32)
    select = y_label_hat[:, 0] != y_label_hat[:, 1]

    print("Nonzero in disagreement select: ", tf.math.count_nonzero(select).numpy())

    def print_resume_stats(d: tf.Tensor):
        print(f"\t\tAverage probability distance: {tf.reduce_mean(d)}")
        for q in [99, 90, 25, 10, 1]:
            print(f"\t\t{q}% percentile: {tfp.stats.percentile(d, q)}")

    d = l2_probability_distance(y_hat[select])
    print("\tOn disagreement:")

    if d.shape == 0:
        print("\t\tNo disagreement!")
    else:
        print(
            f"\t\tTotal % of data: {tf.math.count_nonzero(select)} / {y.shape[0]} ({tf.math.count_nonzero(select) / y.shape[0] * 100:.2f}%)"
        )
        for im in [0, 1]:
            acc = tf.math.count_nonzero(
                y_label_hat[select][:, im] == y[select]
            ) / tf.math.count_nonzero(select)
            print(f"\t\tModel {im} accuracy: {acc * 100:.2f}%")
        print_resume_stats(d)

        if print_digits:
            plot_digit_grid(X[select], y[select])
        if print_confusion_matrices:
            render_confusion_matrices(
                X[select], y[select], y_biased[select], y_label_hat[select]
            )

    acc = y_label_hat[~select][:, 0] == y[~select]
    correct = tf.math.count_nonzero(acc)
    total = tf.math.count_nonzero(~select)
    acc = correct / total

    print("\tOn agreement:")
    print(f"\t\tTotal % of data: {tf.math.count_nonzero(~select) / y.shape[0]}")
    print(f"\t\tAccuracy: {acc}")
    d = l2_probability_distance(y_hat[~select])
    print_resume_stats(d)

    if print_digits:
        plot_digit_grid(X[~select], y[~select])
    if print_confusion_matrices:
        render_confusion_matrices(
            X[~select], y[~select], y_biased[~select], y_label_hat[~select], models=[0]
        )


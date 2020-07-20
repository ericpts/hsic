import seaborn as sns
import concurrent.futures
import argparse
import numpy as np
import subprocess
import os
from pathlib import Path
from typing import List, Optional
import json
import pandas as pd
from collections import defaultdict
import yaml
import json
import gin
from typing import Dict, Any
import shutil

_gin_columns_rename = {
    "compute_combined_loss.diversity_loss_coefficient": "lambda",
    "diversity_loss.kernel": "kernel",
    "diversity_loss.independence_measure": "indep",
}


def _parse_gin_config(config: str) -> Dict[str, Any]:
    tokens = config.split()
    n = len(tokens)
    assert n % 3 == 0
    i = 0
    ret = {}
    while i < n:
        key = tokens[i]
        assert tokens[i + 1] == "="
        value = eval(tokens[i + 2])
        ret[key] = value
        i += 3
    return ret


def _read_problem_raw_data(problem: str) -> pd.DataFrame:
    rows = []
    for yaml_config in Path(problem).glob("**/config.yaml"):
        with open(yaml_config, "rt") as f:
            cfg_dict = yaml.load(f, Loader=yaml.CLoader)

        results_json = Path(cfg_dict["results_json_output"])
        if not results_json.exists():
            print(f"Could not find {results_json}! Skipping folder...")
            continue

        with results_json.open("rt") as f:
            results_dict = json.load(f)

        gin_config = _parse_gin_config(Path(cfg_dict["gin_config_file"]).read_text())
        row = {**results_dict, **gin_config}
        row["original_config"] = yaml_config

        if "diversity_loss_coefficient":
            del row["diversity_loss_coefficient"]

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


def read_problem(problem: str) -> pd.DataFrame:
    DF = _read_problem_raw_data("toy")
    DF["weights"] = DF["weights"].apply(_weights_to_numpy)
    DF[["cos", "norm"]] = DF["weights"].apply(_process_weights_for_cos_and_norm)
    return DF

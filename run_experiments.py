#!/usr/bin/env python3
import os
import concurrent.futures
import argparse
import numpy as np
import subprocess
import os
from pathlib import Path
from typing import List
from tqdm import tqdm
import lib_biased_mnist

LABEL_CORRELATIONS = [0.999, 0.997, 0.995, 0.99]


def make_biased_mnist_runs(
    indep: str, k: str, l: float, corr: float, model: str, runs: int,
) -> List[Path]:
    root = Path(
        f"biased_mnist/{indep}/{model}/label_correlation_{corr}/{k}_kernel/lambda_{l}"
    )
    ret = []
    r_0 = -1
    already_ran = True
    while already_ran:
        r_0 += 1
        already_ran = (root / f"run_{r_0}" / "results.json").exists()

    for r in range(r_0, runs):
        d = root / f"run_{r}"
        os.makedirs(d, exist_ok=True)

        assert not (d / "results.json").exists()

        gin_config = d / "config.gin"
        gin_config.write_text(
            f"""
diversity_loss.independence_measure = '{indep}'
diversity_loss.kernel = '{k}'
compute_combined_loss.diversity_loss_coefficient = {l}
BiasedMnistProblem.training_data_label_correlation = {corr}
BiasedMnistProblem.filter_for_digits = [0, 1]
    """
        )

        results_json = d / "results.json"
        yaml_config = d / "config.yaml"
        yaml_config.write_text(
            f"""
gin_config_file: {str(gin_config)}
results_json_output: {str(results_json)}
epochs: 100
problem: biased_mnist
"""
        )
        ret.append(yaml_config)
    return ret


def generate_biased_mnist_configs() -> List[Path]:
    """ Returns a list of configs to run. """
    lambdas = [
        0,
        1,
        4,
        64,
        128,
    ]
    models = ["mlp"]  # , "cnn"]
    kernels = ["rbf"]
    runs = 1

    ret = []
    for indep in ["conditional_hsic", "unbiased_hsic"]:
        for k in kernels:
            for l in lambdas:
                for corr in LABEL_CORRELATIONS:
                    for m in models:
                        ret.extend(make_biased_mnist_runs(indep, k, l, corr, m, runs))
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_processes", type=int, default=16)
    args = parser.parse_args()
    lib_biased_mnist.regenerate_all_data(LABEL_CORRELATIONS)
    configs = generate_biased_mnist_configs()

    total_configs = len(configs)
    tasks = []
    print("Generated configs.")

    with concurrent.futures.ThreadPoolExecutor(args.n_processes) as exe:
        for ic, c in enumerate(configs):
            env = os.environ
            env["TF_CPP_MIN_LOG_LEVEL"] = "2"
            if ic % args.n_processes != 0:
                env["CUDA_VISIBLE_DEVICES"] = ""
            sp_args = ["python3", "main.py", "--yaml_config_file", str(c)]
            tasks.append(exe.submit(subprocess.run, sp_args, check=True, env=env))
        print("Launched tasks.")
        for t in tqdm(concurrent.futures.as_completed(tasks)):
            pass
    print("Finished all tasks.")


if __name__ == "__main__":
    main()

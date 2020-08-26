#!/usr/bin/env python3
import os
import concurrent.futures
import argparse
import subprocess
import os
from pathlib import Path
from typing import List
from tqdm import tqdm
import lib_biased_mnist
import mlflow
import lib_mlflow

LABEL_CORRELATIONS = [1, 0.999]
NOISE_LEVELS = [0, 50, 100, 150]


def make_biased_mnist_runs(
    experiment_name: str,
    indep: str,
    k: str,
    l: float,
    corr: float,
    model: str,
    runs: int,
    noise_level: int,
    initial_lr: float,
    n_digits: int,
) -> List[Path]:
    root = (
        Path(os.environ["SCRATCH"])
        / experiment_name
        / f"biased_mnist"
        / f"{n_digits}_digits"
        / f"{indep}"
        / f"{model}"
        / f"label_correlation_{corr}"
        / f"{k}_kernel"
        / f"lambda_{l}"
        / f"noise_{noise_level}"
        / f"initial_lr{initial_lr}"
    ).resolve()

    ret = []

    for r in range(runs):
        d = root / f"run_{r}"
        os.makedirs(d, exist_ok=True)

        if (d / "results.json").exists():
            continue

        gin_config = d / "config.gin"
        gin_config.write_text(
            f"""
diversity_loss.independence_measure = '{indep}'
diversity_loss.kernel = '{k}'
compute_combined_loss.diversity_loss_coefficient = {l}
BiasedMnistProblem.training_data_label_correlation = {corr}
BiasedMnistProblem.filter_for_digits = {list(range(n_digits))}
BiasedMnistProblem.model_type = 'mlp'
BiasedMnistProblem.background_noise_level = {noise_level}
Problem.n_epochs = 100
Problem.initial_lr = {initial_lr}
Problem.decrease_lr_at_epochs = [20, 40, 80]
Problem.n_models = 2
Problem.optimizer=@tf.keras.optimizers.Nadam
tf.keras.optimizers.Nadam.epsilon = 0.001
get_weight_regularizer.strength = 0.01
    """
        )

        results_json = d / "results.json"
        yaml_config = d / "config.yaml"
        models_output_dir = d / "models"
        yaml_config.write_text(
            f"""
gin_config_file: {str(gin_config)}
results_json_output: {str(results_json)}
models_output_dir: {str(models_output_dir)}
problem: biased_mnist
experiment_name: {experiment_name}
"""
        )
        ret.append(yaml_config)
    return ret


def generate_biased_mnist_configs(experiment_name: str) -> List[Path]:
    """ Returns a list of configs to run. """
    lambdas = [
        0,
        1 / 32,
        1 / 4,
        1 / 2,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
    ]
    models = ["mlp"]
    kernels = ["rbf"]
    runs = 3

    ret = []
    for initial_lr in [0.01]:
        for noise_level in NOISE_LEVELS:
            for indep in ["conditional_cka", "cka"]:
                for k in kernels:
                    for l in lambdas:
                        for corr in LABEL_CORRELATIONS:
                            for m in models:
                                ret.extend(
                                    make_biased_mnist_runs(
                                        experiment_name,
                                        indep,
                                        k,
                                        l,
                                        corr,
                                        m,
                                        runs,
                                        noise_level,
                                        initial_lr,
                                        n_digits=10,
                                    )
                                )
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_processes", type=int, default=16)
    parser.add_argument("--experiment_name", type=str, required=True)
    args = parser.parse_args()
    lib_biased_mnist.regenerate_all_data(LABEL_CORRELATIONS, NOISE_LEVELS)
    configs = generate_biased_mnist_configs(args.experiment_name)

    lib_mlflow.cluster_setup()
    lib_mlflow.set_remote_eth_server()
    if not mlflow.get_experiment_by_name(args.experiment_name):
        mlflow.create_experiment(args.experiment_name)

    tasks = []
    print(f"Generated {len(configs)} configs.")

    with concurrent.futures.ThreadPoolExecutor(args.n_processes) as exe:
        for ic, c in enumerate(configs):
            env = os.environ
            env["TF_CPP_MIN_LOG_LEVEL"] = "2"
            if ic % args.n_processes != 0:
                env["CUDA_VISIBLE_DEVICES"] = ""
            sp_args = ["python3", "main.py", "--yaml_config_file", str(c)]
            tasks.append(exe.submit(subprocess.run, sp_args, check=True, env=env,))
        print("Launched tasks.")
        for t in tqdm(concurrent.futures.as_completed(tasks)):
            pass
    print("Finished all tasks.")


if __name__ == "__main__":
    main()

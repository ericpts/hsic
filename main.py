import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import gin
import argparse
import yaml
from datetime import datetime
from pathlib import Path
import json
from typing import List, Tuple, Dict
import lib_toy
import lib_biased_mnist
import lib_celeb_a
import lib_analysis
import mlflow


PROBLEM_DICT = {
    "toy": lib_toy.ToyProblem,
    "biased_mnist": lib_biased_mnist.BiasedMnistProblem,
    "celeb_a": lib_celeb_a.CelebAProblem,
}


def main(config: Dict, gin_overrides: List[str]):
    server = tf.profiler.experimental.server.start(6009)

    assert "gin_config_file" in config
    assert "results_json_output" in config
    assert "problem" in config
    assert config["problem"] in list(
        PROBLEM_DICT.keys()
    ), f'Expected problem to be one of {list(PROBLEM_DICT.keys())}; found {config["problem"]}'

    gin.parse_config_files_and_bindings([config["gin_config_file"]], gin_overrides)
    problem = PROBLEM_DICT[config["problem"]]()

    if "MLFLOW_REMOTE_SERVER" in os.environ:
        print("Using remote MLFlow server")
        assert "MLFLOW_USERNAME" in os.environ, "Please provide mlflow username"
        assert "MLFLOW_PASSWORD" in os.environ, "Please provide mlflow password"

        mlflow_server = os.environ["MLFLOW_REMOTE_SERVER"]
        mlflow_username = os.environ["MLFLOW_USERNAME"]
        mlflow_password = os.environ["MLFLOW_PASSWORD"]

        mlflow.set_tracking_uri(
            f"https://{mlflow_username}:{mlflow_password}@{mlflow_server}"
        )

    with mlflow.start_run():
        results, models = problem.train()
        mlflow.log_params(lib_analysis._parse_gin_config(gin.operative_config_str()))
        mlflow.log_params(config)
        return results, models


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_config_file",
        type=str,
        help="Yaml config file containing all parameters",
        required=True,
    )
    parser.add_argument(
        "--gin_override",
        type=str,
        help="Override gin parameters",
        nargs="*",
        action="append",
    )
    args = parser.parse_args()
    assert Path(args.yaml_config_file).exists
    config = yaml.load(Path(args.yaml_config_file).read_text(), Loader=yaml.FullLoader)

    gin_overrides = []
    if args.gin_override:
        for opt in args.gin_override:
            gin_overrides.extend(opt)

    results, models = main(config, gin_overrides)
    model_paths = []
    for im, m in enumerate(models):
        out = Path(config["models_output_dir"]) / f"model-{im}.h5"
        out.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out))
        model_paths.append(str(out))
    results["model_paths"] = model_paths
    with open(config["results_json_output"], "w+t") as f:
        json.dump(results, f)


if __name__ == "__main__":
    cli_main()

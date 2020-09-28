#!/usr/bin/env python3
import os
import gin
import argparse
import yaml
from pathlib import Path
import json
from typing import Dict, List
import lib_toy
import lib_biased_mnist
import lib_celeb_a
import lib_waterbirds
import lib_analysis
import gin.tf.external_configurables
import lib_mlflow
import mlflow
import lib_problem
import tensorflow as tf


@gin.configurable()
def main(experiment_name: str):
    print("Using remote MLFlow server")
    lib_mlflow.set_remote_eth_server()

    print(f"Using experiment name {experiment_name}")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        base_dir = Path(os.environ["SCRATCH"]) / experiment_name
        lib_mlflow.log_param("run_id", run.info.run_id)
        problem = lib_problem.Problem(base_dir)

        results, models = problem.train()

        gin_config_string = gin.operative_config_str()
        lib_mlflow.log_params(lib_analysis._parse_gin_config(gin_config_string))
        # Include also all the default parameters in the final gin config file.
        (base_dir / "full_config.gin").write_text(gin_config_string)

        print("Saving models")
        model_base_save_path = Path(os.environ["SCRATCH"]) / experiment_name / "models"
        model_base_save_path.mkdir(parents=True, exist_ok=True)
        for im, m in enumerate(models):
            out = model_base_save_path / f"model-{im}.h5"
            m.save_weights(str(out))
        results["model_base_save_path"] = model_base_save_path

    print(results)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gin_config_file",
        type=str,
        help="Gin config file containing all parameters",
        required=True,
    )
    parser.add_argument(
        "--gin_override", type=str, help="Override gin parameters", action="append"
    )
    args = parser.parse_args()
    for config_path in args.gin_config_file:
        assert Path(config_path).exists

    gin.parse_config_files_and_bindings([args.gin_config_file], args.gin_override)
    main()


if __name__ == "__main__":
    cli_main()

#!/usr/bin/env python3
import shutil
import tensorflow as tf
import os
import gin
import argparse
from pathlib import Path
import lib_analysis
import gin.tf.external_configurables
import lib_mlflow
import mlflow
import lib_problem


@gin.configurable()
def main(experiment_name: str):
    print("Using remote MLFlow server")
    lib_mlflow.setup()

    print(f"Using experiment name {experiment_name}")
    lib_mlflow.try_until_success(mlflow.set_experiment, experiment_name)

    run = lib_mlflow.try_until_success(mlflow.start_run)

    run_id = run.info.run_id
    lib_mlflow.log_param("run_id", run.info.run_id)
    base_dir = Path(os.environ["SCRATCH"]) / experiment_name / run_id
    problem = lib_problem.Problem(base_dir=base_dir)
    results, models = problem.train()

    gin_config_string = gin.operative_config_str()
    lib_mlflow.log_params(lib_analysis._parse_gin_config(gin_config_string))
    # Include also all the default parameters in the final gin config file.
    (base_dir / "full_config.gin").write_text(gin_config_string)

    print("Saving models")
    model_base_save_path = base_dir / "models"
    model_base_save_path.mkdir(parents=True, exist_ok=True)
    for im, m in enumerate(models):
        out = model_base_save_path / f"model-{im}"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir()
        tf.keras.models.save_model(m, str(out))
    results["model_base_save_path"] = model_base_save_path

    print(results)

    lib_mlflow.shutdown()
    lib_mlflow.try_until_success(mlflow.end_run)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gin_file",
        type=str,
        help="Gin config file containing all parameters",
        required=True,
    )
    parser.add_argument(
        "--gin_param", type=str, help="Override gin parameters", action="append"
    )
    args = parser.parse_args()
    for config_path in args.gin_file:
        assert Path(config_path).exists

    gin.parse_config_files_and_bindings([args.gin_file], args.gin_param)

    main()


if __name__ == "__main__":
    cli_main()

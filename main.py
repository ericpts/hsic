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


PROBLEM_DICT = {
    "toy": lib_toy.ToyProblem,
    "biased_mnist": lib_biased_mnist.BiasedMnistProblem,
    "celeb_a": lib_celeb_a.CelebAProblem,
    "waterbirds": lib_waterbirds.WaterbirdsProblem,
}


def main(config: Dict, gin_overrides: List[str]):

    assert "gin_config_file" in config
    assert "results_json_output" in config
    assert "problem" in config
    assert config["problem"] in list(
        PROBLEM_DICT.keys()
    ), f'Expected problem to be one of {list(PROBLEM_DICT.keys())}; found {config["problem"]}'

    gin.parse_config_files_and_bindings([config["gin_config_file"]], gin_overrides)
    problem = PROBLEM_DICT[config["problem"]]()

    results, models = problem.train()

    gin_config_string = gin.operative_config_str()

    # Include also all the default parameters in the final gin config file.
    Path(config["gin_config_file"]).write_text(gin_config_string)

    lib_mlflow.log_params(lib_analysis._parse_gin_config(gin_config_string))
    lib_mlflow.log_params(config)
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

    print("Using remote MLFlow server")
    lib_mlflow.set_remote_eth_server()

    assert "experiment_name" in config
    experiment_name = config["experiment_name"]
    print(f"Using experiment name {experiment_name}")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        lib_mlflow.log_param("run_id", run.info.run_id)
        results, models = main(config, gin_overrides)

        print("Saving models")
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

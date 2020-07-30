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


def main(config: Dict, gin_overrides: List[str]):
    assert "gin_config_file" in config
    assert "results_json_output" in config
    assert "problem" in config
    assert config["problem"] in [
        "toy",
        "biased_mnist",
    ], f'Expected problem to be one of toy or biased_mnist; found {config["problem"]}'

    gin.parse_config_files_and_bindings([config["gin_config_file"]], gin_overrides)

    if config["problem"] == "toy":
        problem = lib_toy.ToyProblem()
    elif config["problem"] == "biased_mnist":
        problem = lib_biased_mnist.BiasedMnistProblem()
    else:
        raise ValueError(f'Unexpected problem: {config["problem"]}')

    results, models = problem.train()

    model_paths = []
    for im, m in enumerate(models):
        out = Path(config["models_output_dir"]) / f"model-{im}.h5"
        out.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out))
        model_paths.append(str(out))
    results["model_paths"] = model_paths

    with open(config["results_json_output"], "w+t") as f:
        json.dump(results, f)


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
    for opt in args.gin_override:
        gin_overrides.extend(opt)
    return main(config, gin_overrides)


if __name__ == "__main__":
    cli_main()

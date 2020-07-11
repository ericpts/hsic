#!/usr/bin/env python3
import concurrent.futures
import argparse
import numpy as np
import subprocess
import os
from pathlib import Path
from typing import List
from tqdm import tqdm


def generate_configs(problem: str) -> List[Path]:
    """ Returns a list of configs to run. """
    lambdas = [
        0,
        1 / 16,
        1 / 8,
        1 / 4,
        1 / 2,
        1,
        2,
        4,
        16,
        64,
        256,
        1024,
        4096,
        8192,
        16348,
        32768,
        65536,
    ]
    kernels = ["linear", "rbf"]
    runs = 10

    ret = []
    for indep in ["hsic", "cka"]:
        root = Path(f"{problem}/{indep}")
        for k in kernels:
            for l in lambdas:

                r_0 = -1
                already_ran = True
                while already_ran:
                    r_0 += 1
                    already_ran = (
                        root
                        / f"{k}_kernel"
                        / f"lambda_{l}"
                        / f"run_{r_0}"
                        / "results.json"
                    ).exists()

                for r in range(r_0, r_0 + runs):
                    d = root / f"{k}_kernel" / f"lambda_{l}" / f"run_{r}"
                    os.makedirs(d)

                    assert not (d / "results.json").exists()

                    gin_config = d / "config.gin"
                    gin_config.write_text(
                        f"""
diversity_loss.independence_measure = '{indep}'
diversity_loss.kernel = '{k}'
compute_combined_loss.diversity_loss_coefficient = {l}
                    """
                    )

                    results_json = d / "results.json"
                    yaml_config = d / "config.yaml"
                    yaml_config.write_text(
                        f"""
gin_config_file: {str(gin_config)}
results_json_output: {str(results_json)}
epochs: 20
problem: {problem}
        """
                    )
                    ret.append(yaml_config)
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem", type=str, choices=["toy", "biased_mnist"], required=True
    )

    args = parser.parse_args()
    configs = generate_configs(args.problem)
    total_configs = len(configs)
    tasks = []
    print("Generated configs.")
    with concurrent.futures.ThreadPoolExecutor(32) as exe:
        for c in configs:
            args = ["python3", "main.py", "--yaml_config_file", str(c)]
            tasks.append(exe.submit(subprocess.run, args, check=True))
        print("Launched tasks.")
        for t in tqdm(concurrent.futures.as_completed(tasks)):
            pass
    print("Finished all tasks.")


if __name__ == "__main__":
    main()

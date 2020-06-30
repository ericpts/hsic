#!/usr/bin/env python3
import argparse
import subprocess
import os
from pathlib import Path
from typing import List


def generate_configs() -> List[Path]:
""" Returns a list of configs to run. """
    lambdas = range(0, 1, 0.05)
    kernels = ["linear", "rbf"]

    ret = []
    for k in kernels:
        root = Path(f"{k}_kernel")
        os.makedirs(root)
        for l in lambas:
            d = root / f"lambda_{l}"
            os.mkdir(d)
            gin_config = d / "config.gin"
            gin_config.write_text(
                f"""
diversity_loss.kernel = '{k}'
compute_combined_loss.diversity_loss_coefficient = {l}
            """
            )

            yaml_cofig = d / "config.yaml"
            yaml_config.write_text(
                    f"""
gin_config_file: {str(gin_config)}
results_json_output: results.json
""")
            ret.append(yaml_config)
    return ret


def main():
    configs = generate_configs()
    for c in configs:
        args = ['python3', 'toy.py', '--yaml_config_file', str(c)]
        subprocess.run(args, check=True)

if __name__ == '__main__':
    main()

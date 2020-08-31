#!/usr/bin/env python3
from typing import Dict, List, Optional, Callable
from pathlib import Path
import lib_util


class Experiment(object):
    def __init__(
        self,
        name: str,
        problem: str,
        n_runs: int,
        experiment_dict: Dict,
        f_pre_experiment: Optional[Callable] = None,
    ):
        self.name = name
        self.problem = problem
        self.n_runs = n_runs
        self.f_pre_experiment = f_pre_experiment

        for k, v in experiment_dict.items():
            if type(v) == type([]):
                continue
            else:
                experiment_dict[k] = [v]

        self.experiment_dict = experiment_dict

    def on_before_start(self):
        if self.f_pre_experiment:
            self.f_pre_experiment()

    def generate_configs(self, base_dir: Path) -> List:
        all_configs = []

        def once(**kwargs):
            output_lines = []
            for k in sorted(kwargs.keys()):
                v = kwargs[k]
                if type(v) == type(""):
                    # Special gin type.
                    if v[0] == "@":
                        output_lines.append(f"{k}: {v}")
                    else:
                        output_lines.append(f"{k}: '{v}'")
                else:
                    output_lines.append(f"{k}: {v}")

            root = Path(base_dir)
            for k in sorted(kwargs.keys()):
                v = kwargs[k]
                root = root / f"{k}_{v}"
            root = root.resolve()

            current_configs = []
            for r in range(self.n_runs):
                current_run = root / f"run_{r}"
                os.makedirs(current_run, exist_ok=True)
                if (current_run / "results.json").exists():
                    continue

                gin_config = current_run / "config.gin"
                gin_config.write_text("\n".join(output_lines))

                results_json = current_run / "results.json"
                yaml_config = current_run / "config.yaml"
                models_output_dir = current_run / "models"

                yaml_config.write_text(
                    f"""
gin_config_file: {str(gin_config)}
results_json_output: {str(results_json)}
models_output_dir: {str(models_output_dir)}
problem: {self.problem}
experiment_name: {self.name}
        """
                )
                current_configs.append(yaml_config)
            all_configs.extend(current_configs)

        lib_util.with_grid_values(self.experiment_dict, once)
        return configs

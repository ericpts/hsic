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
import importlib
from experiments.lib_experiment import Experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_processes", type=int, default=16)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    args = parser.parse_args()

    experiment_module = importlib.import_module(f"experiments.{args.experiment}")
    experiment = experiment_module.experiment  # type: Experiment
    experiment.on_before_start()
    configs = experiment.generate_configs(Path(args.base_dir))
    print(f"Generated {len(configs)} configs.")

    experiment_name = experiment.name

    lib_mlflow.cluster_setup()
    lib_mlflow.set_remote_eth_server()
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)

    tasks = []

    with concurrent.futures.ThreadPoolExecutor(args.n_processes) as exe:
        err = None
        try:
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

        except KeyboardInterrupt as e:
            print("Stopping due to SIGINT")
            exe.shutdown(wait=False)
            err = e
        except Exception as e:
            print(f"Got exception from a subprocess; terminating...")
            exe.shutdown(wait=False)
            err = e

        if err is not None:
            raise err

    print("Finished all tasks.")


if __name__ == "__main__":
    main()


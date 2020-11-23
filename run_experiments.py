#!/usr/bin/env python3
import argparse
import itertools
import lib_jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument(
        "-wp",
        "--with_param",
        type=str,
        action="append",
        default=[],
        help="Optional repeated argument of the form k=[v], "
        "will be included in the cartesian product of parameters, "
        "using k as the gin parameter name. "
        "Example usage: --with_param data.batch_size=[32,64,128]",
    )
    args = parser.parse_args()

    params = {}
    for param_string in args.with_param:
        [k, v] = [s.strip() for s in param_string.split("=")]
        if not v.startswith("@"):
            v = eval(v)  # Risky but what you gonna do about it.

        if type(v) == type([]):
            pass
        else:
            v = [v]
        params[k] = v

    for config in itertools.product(
        *params.values(),
    ):
        (*gin_args_tuple,) = config
        gin_args = list(zip(params.keys(), gin_args_tuple))
        gin_args.append(("main.experiment_name", args.experiment))
        cli_args = [
            ("gin_file", "configs/biased_mnist.gin"),
        ]

        lib_jobs.launch_bsub(
            nhours=1,
            main_python_file="main.py",
            cli_args=cli_args,
            gin_args=gin_args,
            n_cpus=2,
            require_gpu=False,
            job_name=f"log_{args.experiment}",
        )


if __name__ == "__main__":
    main()

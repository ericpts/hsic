import mlflow
import time
import subprocess
import traceback

MLFLOW_SERVER = "exp-02.mlflow-yang.inf.ethz.ch"
MLFLOW_USERNAME = "exp-02.mlflow-yang.ericst"
MLFLOW_PASSWORD = "parolaeric"


def cluster_setup():
    subprocess.check_call(["zsh", "-c", "source ~/.zshrc && module unload eth_proxy"])


def set_remote_eth_server():
    tracking_uri = f"https://{MLFLOW_USERNAME}:{MLFLOW_PASSWORD}@{MLFLOW_SERVER}"
    mlflow.set_tracking_uri(tracking_uri)


def try_until_success(f):
    done = False
    while not done:
        try:
            f()
            done = True
        except (IOError, mlflow.exceptions.MlflowException) as err:
            print("Caught error when logging to mlflow!")
            traceback.print_exc()
            print("Stack trace: ")
            traceback.print_stack()
            time.sleep(5)


def log_param(k, v):
    try_until_success(lambda: mlflow.log_param(k, v))


def log_params(params):
    try_until_success(lambda: mlflow.log_params(params))


def log_metric(k, v, step):
    try_until_success(lambda: mlflow.log_metric(k, v, step))


def set_tag(k, v):
    try_until_success(lambda: mlflow.set_tag(k, v))

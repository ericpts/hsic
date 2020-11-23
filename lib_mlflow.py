import mlflow
import requests
import time
import subprocess
from tqdm import tqdm
import queue
import atexit
import concurrent.futures

MLFLOW_SERVER = "exp-02.mlflow-yang.inf.ethz.ch"
MLFLOW_USERNAME = "exp-02.mlflow-yang.ericst"
MLFLOW_PASSWORD = "parolaeric"


logging_queue = queue.Queue()
outstanding_logs = []
logging_thread_pool: concurrent.futures.ThreadPoolExecutor = None


def try_until_success(f, *args):
    while True:
        try:
            return f(*args)
        except (
            IOError,
            mlflow.exceptions.MlflowException,
            requests.exceptions.ConnectionError,
        ) as err:
            print("Retrying mlflow...")
            time.sleep(1)


def async_log(f, *args):
    outstanding_logs.append(logging_thread_pool.submit(try_until_success, f, *args))


def setup():
    global logging_thread_pool
    logging_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    tracking_uri = f"https://{MLFLOW_USERNAME}:{MLFLOW_PASSWORD}@{MLFLOW_SERVER}"
    mlflow.set_tracking_uri(tracking_uri)


def shutdown():
    print("Waiting for all logging to finish.")
    for o in tqdm(outstanding_logs):
        o.result()
    logging_thread_pool.shutdown(wait=True)


def cluster_setup():
    subprocess.check_call(["zsh", "-c", "source ~/.zshrc && module unload eth_proxy"])


def log_param(k, v):
    async_log(mlflow.log_param, k, v)


def log_params(params):
    async_log(mlflow.log_params, params)


def log_metric(k, v, step):
    async_log(mlflow.log_metric, k, v, step)


def set_tag(k, v):
    async_log(mlflow.set_tag, k, v)

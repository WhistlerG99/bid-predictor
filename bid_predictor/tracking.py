import os
import time
import threading
import mlflow


class MlflowCallback(object):
    """
    CatBoost callback to log training and validation metrics to MLflow.
    """

    def after_iteration(self, info):
        # Log metrics after each iteration
        iteration = info.iteration
        self.iteration = iteration

        # Log training loss
        train_loss = info.metrics["learn"]["Logloss"][-1]
        mlflow.log_metric("train_loss", train_loss, step=iteration)

        # Log validation loss
        validation_loss = info.metrics["validation"]["Logloss"][-1]
        mlflow.log_metric("validation_loss", validation_loss, step=iteration)

        # Log validation AUC
        validation_auc = info.metrics["validation"]["AUC"][-1]
        mlflow.log_metric("validation_auc", validation_auc, step=iteration)

        return True  # Continue training


def follow_tsv(
    path: str,
    key: str,
    run_id: str,
    client: mlflow.MlflowClient,
    stop_event: threading.Event,
):
    """
    Tail a CatBoost TSV metrics file and log values to MLflow for a specific run.

    Parameters
    ----------
    path : str
        Path to the TSV file (e.g., learn_error.tsv, test_error.tsv).
    key : str
        Metric name to log (e.g., "train_Logloss", "eval_Logloss").
    run_id : str
        Active MLflow run ID to log into.
    client : mlflow.MlflowClient
        Reusable MLflow client.
    stop_event : threading.Event
        Cooperative stop signal for the tailer thread.
    """
    # Wait (a bit) for the file to appear
    t0 = time.time()
    while not os.path.exists(path) and not stop_event.is_set():
        if time.time() - t0 > 300:  # 5-minute safety
            return
        time.sleep(0.2)
    if not os.path.exists(path):
        return

    with open(path, "r") as f:
        # skip header if present
        header_line = f.readline()
        _ = header_line  # not used; discard

        while not stop_event.is_set():
            pos = f.tell()
            line = f.readline()
            if not line:
                time.sleep(0.2)
                f.seek(pos)
                continue

            parts = line.strip().split("\t")
            if not parts:
                continue

            # Typical format: iter \t <metric(s)>
            try:
                step = int(parts[0])
            except Exception:
                continue

            # Log the last numeric column
            for c in reversed(parts[1:]):
                try:
                    val = float(c)
                    client.log_metric(run_id, key, val, step=step)
                    break
                except ValueError:
                    continue


def start_catboost_mlflow_stream(train_dir: str, run_id: str, metric_prefix: str = ""):
    """
    Start background threads that tail CatBoost metric TSVs and log to MLflow.

    Returns
    -------
    stop_stream : callable
        Call to stop threads and join them.
    """
    client = mlflow.tracking.MlflowClient()
    stop_event = threading.Event()
    threads = []

    learn_path = os.path.join(train_dir, "learn_error.tsv")
    test_path = os.path.join(train_dir, "test_error.tsv")

    t1 = threading.Thread(
        target=follow_tsv,
        args=(
            learn_path,
            f"{metric_prefix}train_Logloss" if metric_prefix else "train_Logloss",
            run_id,
            client,
            stop_event,
        ),
        daemon=True,
    )
    t2 = threading.Thread(
        target=follow_tsv,
        args=(
            test_path,
            f"{metric_prefix}eval_Logloss" if metric_prefix else "eval_Logloss",
            run_id,
            client,
            stop_event,
        ),
        daemon=True,
    )
    t1.start()
    t2.start()
    threads.extend([t1, t2])

    def stop_stream():
        stop_event.set()
        for t in threads:
            t.join(timeout=2)

    return stop_stream

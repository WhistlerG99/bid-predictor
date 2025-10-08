import os
import time
import threading
import pandas as pd
import numpy as np
import mlflow
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, ClassifierMixin


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


def follow_tsv(path: str, key: str, run_id: str, client: mlflow.MlflowClient, stop_event: threading.Event):
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
        args=(learn_path, f"{metric_prefix}train_Logloss" if metric_prefix else "train_Logloss", run_id, client, stop_event),
        daemon=True,
    )
    t2 = threading.Thread(
        target=follow_tsv,
        args=(test_path,  f"{metric_prefix}eval_Logloss" if metric_prefix else "eval_Logloss",  run_id, client, stop_event),
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


# def start_catboost_mlflow_stream(train_dir, run_id, metric_prefix=""):
#     client = mlflow.tracking.MlflowClient()
#     stop = threading.Event()

#     def _follow_tsv(path, key):
#         # wait for file to appear
#         t0 = time.time()
#         while not os.path.exists(path) and not stop.is_set():
#             if time.time() - t0 > 300:  # 5 min safety
#                 return
#             time.sleep(0.2)
#         if not os.path.exists(path):
#             return

#         with open(path, "r") as f:
#             _ = f.readline()  # skip header
#             while not stop.is_set():
#                 pos = f.tell()
#                 line = f.readline()
#                 if not line:
#                     time.sleep(0.2)
#                     f.seek(pos)
#                     continue
#                 parts = line.strip().split("\t")
#                 if not parts:
#                     continue
#                 try:
#                     step = int(parts[0])
#                     # last numeric column is the metric value
#                     for c in reversed(parts[1:]):
#                         try:
#                             val = float(c)
#                             client.log_metric(run_id, key, val, step=step)
#                             break
#                         except ValueError:
#                             continue
#                 except Exception:
#                     continue

#     threads = []
#     learn_path = os.path.join(train_dir, "learn_error.tsv")
#     test_path = os.path.join(train_dir, "test_error.tsv")

#     t1 = threading.Thread(target=_follow_tsv, args=(learn_path, f"{metric_prefix}train_logloss"), daemon=True)
#     t2 = threading.Thread(target=_follow_tsv, args=(test_path,  f"{metric_prefix}eval_logloss"),  daemon=True)
#     t1.start()
#     t2.start()
#     threads.extend([t1, t2])

#     def stop_stream():
#         stop.set()
#         for t in threads:
#             t.join(timeout=2)

#     return stop_stream


pre_features = [
    "carrier_code",
    "flight_number",
    "travel_date",
    "item_count",
    "usd_base_amount",
    "seats_available",
    "fare_class",
    "offer_time",
    "multiplier_fare_class",
    "multiplier_loyalty",
    "multiplier_success_history",
    "multiplier_payment_type",
    "upgrade_type",
]

features = [
    "carrier_code",
    "flight_number",
    "travel_date",
    "item_count_grp",
    "usd_base_amount_grp",
    "usd_base_amount_25%",
    "usd_base_amount_50%",
    "usd_base_amount_75%",
    "usd_base_amount_max",
    "num_offers_grp",
    "seats_available_grp",
    "fare_class",
    "offer_time_grp",
    "multiplier_fare_class",
    "multiplier_loyalty",
    "multiplier_success_history",
    "multiplier_payment_type",
]

cat_features = [
    "carrier_code",
    "flight_number",
    "item_count_grp",
    "usd_base_amount_grp",
    "num_offers_grp",
    "seats_available_grp",
    "fare_class",
    "offer_time_grp",
]


def add_group_features(out):
    num_offers_col_name = "num_offers"
    usd_base_amount_max_name = "usd_base_amount_max"
    out = out.drop(
        columns=[num_offers_col_name, usd_base_amount_max_name], errors="ignore"
    )
    out = out.merge(
        out.groupby(
            ["carrier_code", "flight_number", "travel_date", "upgrade_type"],
            observed=True,
        )
        .size()
        .rename(num_offers_col_name)
        .reset_index(),
        on=["carrier_code", "flight_number", "travel_date", "upgrade_type"],
    )
    out = out.merge(
        out.groupby(
            ["carrier_code", "flight_number", "travel_date", "upgrade_type"],
            observed=True,
        )["usd_base_amount"]
        .max()
        .rename(usd_base_amount_max_name)
        .reset_index(),
        on=["carrier_code", "flight_number", "travel_date", "upgrade_type"],
    )
    return out


def bin_features(out):
    amount_bins = [-float("inf")] + list(range(100, 1701, 100)) + [float("inf")]
    amount_labels = (
        ["<" + str(amount_bins[1])]
        + [
            f"{amount_bins[i]}-{amount_bins[i+1]}"
            for i in range(1, len(amount_bins) - 2)
        ]
        + [">" + str(amount_bins[-2])]
    )
    out["usd_base_amount_grp"] = pd.cut(
        out["usd_base_amount"], bins=amount_bins, labels=amount_labels
    )
    offer_time_bins = list(range(0, 31, 1)) + [float("inf")]
    offer_time_labels = list(range(1, 32, 1))
    out["offer_time_grp"] = pd.cut(
        out["offer_time"],
        bins=offer_time_bins,
        labels=offer_time_labels,
    )
    item_count_threshold = 4
    out["item_count_grp"] = out["item_count"].apply(
        lambda x: x if x <= item_count_threshold else item_count_threshold + 1
    )
    out["item_count_grp"] = pd.Categorical(out["item_count_grp"])
    num_offers_threshold = 15
    out["num_offers_grp"] = out["num_offers"].apply(
        lambda x: x if x <= num_offers_threshold else num_offers_threshold + 1
    )
    out["num_offers_grp"] = pd.Categorical(out["num_offers_grp"])
    seats_available_low_threshold = -1
    seats_available_high_threshold = 30
    out["seats_available_grp"] = out["seats_available"].apply(
        lambda x: (
            (
                x
                if x < seats_available_high_threshold or np.isnan(x)
                else seats_available_high_threshold
            )
            if x > seats_available_low_threshold or np.isnan(x)
            else seats_available_low_threshold
        )
    )
    out["seats_available_grp"] = pd.Categorical(
        out.fillna({"seats_available_grp": -2}).seats_available_grp.astype(int)
    )
    return out


def quantiles_vectorized(vals, qs=(0.25, 0.50, 0.75)):
    vals = np.asarray(vals)
    n = vals.size
    out = np.full((len(qs), n), np.nan, dtype=float)
    if n <= 1:
        return out

    # Stable sort to keep deterministic ranks with ties
    order = np.argsort(vals, kind="mergesort")
    S = vals[order]                           # sorted values
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)               # rank of each original element

    m = n - 1
    for qi, q in enumerate(qs):
        h = q * (m - 1)                       # NumPy's 'linear' method position
        j = int(np.floor(h))
        gamma = h - j
        j2 = min(j + 1, m - 1)                # clamp upper neighbor inside [0, m-1]

        # Map indices from T (length m) back to S (length n), accounting for the removed item
        idx_a = j  + (j  >= ranks)
        idx_b = j2 + (j2 >= ranks)

        out[qi, :] = (1.0 - gamma) * S[idx_a] + gamma * S[idx_b]

    return out  # shape (len(qs), n)


def quantiles_group(group, col):
    arr = group[col].to_numpy()
    q25, q50, q75 = quantiles_vectorized(arr, (0.25, 0.50, 0.75))
    return pd.DataFrame(
        {
            "q25_excl_self": q25,
            "median_excl_self": q50,
            "q75_excl_self": q75,
        },
        index=group.index,
    )


def add_quantiles(out):
    out[["usd_base_amount_25%", "usd_base_amount_50%", "usd_base_amount_75%"]] = (
        out.groupby(
            ["carrier_code", "flight_number", "travel_date", "upgrade_type"],
            group_keys=False,
            observed=True,
            sort=False,
        ).apply(quantiles_group, col="usd_base_amount", include_groups=False)
    )
    return out

# def loo_quantiles(group, col):
#     vals = group[col].to_numpy()
#     n = len(vals)
#     q25, q50, q75 = [], [], []
#     for i in range(n):
#         if n == 1:
#             q25.append(np.nan)
#             q50.append(np.nan)
#             q75.append(np.nan)
#         else:
#             others = np.concatenate([vals[:i], vals[i + 1 :]])
#             q25.append(np.quantile(others, 0.25))
#             q50.append(np.quantile(others, 0.50))
#             q75.append(np.quantile(others, 0.75))
#     return pd.DataFrame(
#         {"q25_excl_self": q25, "median_excl_self": q50, "q75_excl_self": q75},
#         index=group.index,
#     )


# def add_loo_quantiles(out):
#     out[["usd_base_amount_25%", "usd_base_amount_50%", "usd_base_amount_75%"]] = (
#         out.groupby(
#             ["carrier_code", "flight_number", "travel_date", "upgrade_type"],
#             group_keys=False,
#             observed=True,
#         ).apply(loo_quantiles, col="usd_base_amount", include_groups=False)
#     )
#     return out


# --- Wrappers around your existing funcs so they can be used in pipelines ---
def add_group_features_wrapper(X):
    if isinstance(X, pd.DataFrame):
        return add_group_features(X)
    else:
        return X


def bin_features_wrapper(X):
    if isinstance(X, pd.DataFrame):
        return bin_features(X)
    else:
        return X


def add_quantiles_wrapper(X):
    if isinstance(X, pd.DataFrame):
        return add_quantiles(X)
    else:
        return X


def reduce_features(X):
    if isinstance(X, pd.DataFrame):
        return X[features]
    else:
        return X


# FunctionTransformer allows arbitrary pandas-based functions
group_features_transformer = FunctionTransformer(add_group_features_wrapper)
bin_features_transformer = FunctionTransformer(bin_features_wrapper)
quantiles_transformer = FunctionTransformer(add_quantiles_wrapper)
reduce_features_transformer = FunctionTransformer(reduce_features)


def get_train_dir():
    # Prefer CATBOOST_TRAIN_DIR if present, else /opt/ml/output
    train_dir = os.environ.get("CATBOOST_TRAIN_DIR", "/opt/ml/output/catboost")
    try:
        os.makedirs(train_dir, exist_ok=True)
    except PermissionError:
        # Fall back to /tmp to avoid failing the whole job
        train_dir = os.path.join("/tmp", "catboost")
        os.makedirs(train_dir, exist_ok=True)
        print(f"WARNING: /opt/ml/output not writable; using {train_dir}")
    return train_dir


# ---- 1) Minimal routing-aware wrapper
class CBC(BaseEstimator, ClassifierMixin):
    def __init__(self, **cb_params):
        train_dir = get_train_dir()
        cb_params["train_dir"] = train_dir
        cb_params["allow_writing_files"] = True
        cb_params["verbose"] = 1
        self.cb_params = cb_params
        self._cb = None

    # sklearn will route eval_set here if we request it on the instance
    def fit(self, X, y=None, eval_set=None, **fit_kwargs):


        self._cb = CatBoostClassifier(
            **self.cb_params
        )
        # eval_set supports (X_val, y_val) tuples or Pool objects
        self._cb.fit(
            X,
            y,
            eval_set=eval_set,
            cat_features=cat_features,
            # callbacks=[
            #     MlflowCallback()
            # ],
            ** fit_kwargs,
        )
        return self

    def __sklearn_is_fitted__(self):
        if hasattr(self, "_cb") and self._cb is not None:
            return self._cb.is_fitted()

    # pass-through predict/predict_proba
    def predict(self, X):
        return self._cb.predict(X)

    def predict_proba(self, X):
        return self._cb.predict_proba(X)

    def get_feature_importance(self, *args, **kwargs):
        return self._cb.get_feature_importance(*args, **kwargs)

    # make params grid-searchable
    def get_params(self, deep=True):
        return self.cb_params.copy()

    def set_params(self, **params):
        self.cb_params.update(params)
        return self


def build_pipeline(**kw):
    # CatBoostClassifier integrates with sklearn API
    clf = CBC(
        task_type=kw.get("task_type", "CPU"),
        devices=kw.get("devices", "0"),
        loss_function="Logloss",
        eval_metric="AUC",
        random_state=42,
        auto_class_weights="Balanced",
    ).set_fit_request(eval_set=True)

    pipeline = Pipeline(
        steps=[
            ("group", group_features_transformer),
            ("bin", bin_features_transformer),
            ("loo", quantiles_transformer),
            ("reduce", reduce_features_transformer),
            ("clf", clf),
        ],
        transform_input=["eval_set"],
    )
    return pipeline

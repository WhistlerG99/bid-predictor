# inference.py
"""
SageMaker SKLearn inference entrypoint with Parquet/CSV/JSON I/O.
Compatible with:
  - Real-time inference endpoints (Model + Endpoint in Pipelines UI)
  - Batch Transform jobs
  - SKLearnModel / Model step in SageMaker Pipelines (SDK or UI)

How to package:
  model.tar.gz
  ├─ model.joblib              # REQUIRED (your trained estimator or pipeline)
  ├─ columns.json              # OPTIONAL, list of feature column names in order
  └─ code/
     ├─ inference.py           # THIS FILE
     └─ requirements.txt       # OPTIONAL: e.g., pandas, pyarrow, joblib

Supported inputs (ContentType):
  - "application/x-parquet"     -> Parquet table (arrow)
  - "text/csv"                  -> CSV (no header assumed unless configured)
  - "application/json"          -> Either:
                                   {"data":[{col:value,...}, ...]}  # rows as dicts
                                   {"data":[[...], [...]]}          # 2D array
                                   [[...], [...]]                   # 2D array
                                   [{"c1":v1, ...}, ...]            # rows as dicts

Outputs (Accept):
  - Defaults to application/json: {"predictions":[...]} or
    {"predictions":[[class_probs...], ...]} when PREDICT_PROBA=1.
  - If Accept == "application/x-parquet", returns a Parquet table with
    a single column "prediction" (or one column per class if proba).
"""
import sys
sys.path.append("/opt/ml/code")
sys.path.append("/opt/ml/model")
sys.path.append("/opt/ml/code/bid_predictor")
sys.path.append("/opt/ml/model/bid_predictor")
import io
import os
import json
import logging
from typing import Any, Tuple, Optional, List, Union

import joblib
import numpy as np
import pandas as pd

# NEW: Flask app for custom container serving
from flask import Flask, request

app = Flask(__name__)

# Parquet is optional but recommended. If not present and client sends Parquet, we'll raise clearly.
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PARQUET = True
except Exception:  # pragma: no cover
    _HAS_PARQUET = False

# ---------- Logging ----------
LOGGER = logging.getLogger("sagemaker-inference")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

# ---------- Env config ----------
# If set (e.g., "1" or "true"), use predict_proba when available
USE_PROBA = os.environ.get("PREDICT_PROBA", "").lower() in {"1", "true", "yes"}
# If your CSV payloads include a header row
CSV_HAS_HEADER = os.environ.get("CSV_HAS_HEADER", "").lower() in {"1", "true", "yes"}
# Coerce all columns to numeric (errors='ignore' keeps non-numeric as-is)
COERCE_NUMERIC = os.environ.get("COERCE_NUMERIC", "1").lower() in {"1", "true", "yes"}

# ---------- Helpers ----------
def _to_json_payload(obj: Any) -> Tuple[str, str]:
    if isinstance(obj, (np.ndarray,)):
        out = obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        out = obj.to_dict(orient="records")
    else:
        out = obj
    return json.dumps({"predictions": out}), "application/json"


def _parquet_response_from_array_or_df(pred: Union[np.ndarray, pd.DataFrame]) -> Tuple[bytes, str]:
    if not _HAS_PARQUET:
        raise RuntimeError("pyarrow is not installed in the inference image to return Parquet.")
    if isinstance(pred, np.ndarray):
        if pred.ndim == 1:
            df = pd.DataFrame({"prediction": pred})
        else:
            df = pd.DataFrame(pred)
    elif isinstance(pred, pd.DataFrame):
        df = pred
    else:
        df = pd.DataFrame({"prediction": pred})
    table = pa.Table.from_pandas(df, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue(), "application/x-parquet"


# ---------- SageMaker-style entrypoints (reused by Flask) ----------
def model_fn(model_dir: str) -> dict:
    """
    Load artifacts from /opt/ml/model.
    Expects pipeline.joblib. Returns dict with model + pre_features.
    """
    model_path = os.path.join(model_dir, "pipeline.joblib")
    model = joblib.load(model_path)

    pre_features = model.feature_config["pre_features"]

    state = {
        "model": model,
        "pre_features": pre_features,
    }
    return state


def predict_fn(data: Union[pd.DataFrame, np.ndarray], model_state: dict) -> Any:
    cols = [
        'offer_id',
        'partner_id',
        'product_id',
        'conf_num',
        'carrier_code',
        'flight_number',
        'departure_timestamp',
        'origination_code',
        'destination_code',
        'days_before_departure',
        'seats_available',
        'item_count',
        'usd_base_amount',
        'fare_class',
        'created_timestamp',
        'offer_time',
        'multiplier_fare_class',
        'multiplier_loyalty',
        'multiplier_success_history',
        'multiplier_payment_type',
        'upgrade_type',
        'from_cabin',
        "usd_base_amount_25%",
        "usd_base_amount_50%",
        "usd_base_amount_75%",
        "usd_base_amount_max",
        "num_offers",
        "bid_rank",
        "acceptance_prob",
        "accept_prob_timestamp",
        "file_timestamp",
    ]

    cols_derived = [
        "days_before_departure",
        "usd_base_amount_25%",
        "usd_base_amount_50%",
        "usd_base_amount_75%",
        "usd_base_amount_max",
        "num_offers",
        "bid_rank",
    ]

    data = data.reset_index(drop=True)
    pre_features = model_state["pre_features"]
    model = model_state["model"]    
    if "conf_num" not in data:
        data["conf_num"] = ""

    X = data[pre_features]
    probs, X_tf = model.transform_and_predict_proba(X)
    data = pd.concat((data, X_tf[cols_derived]), axis=1)

    data["acceptance_prob"] = probs[:,1]
    data["accept_prob_timestamp"] = pd.Timestamp.now()

    # data = data[cols].rename({"usd_base_amount_25%": "usd_base_amount_25_percent",
    #                           "usd_base_amount_50%": "usd_base_amount_50_percent",
    #                           "usd_base_amount_75%": "usd_base_amount_75_percent"})

    return data

def input_fn(request_body: bytes, content_type: str) -> Union[pd.DataFrame, np.ndarray]:
    """
    Convert the request payload to a Pandas DataFrame (preferred) or NumPy array.
    """
    ct = (content_type or "application/x-parquet").split(";")[0].strip().lower()

    if ct == "application/x-parquet":
        if not _HAS_PARQUET:
            raise ValueError("Received Parquet but pyarrow is not installed in the image.")
        table = pq.read_table(io.BytesIO(request_body))
        df = table.to_pandas()
        return df

    if ct == "text/csv":
        if CSV_HAS_HEADER:
            df = pd.read_csv(io.BytesIO(request_body))
        else:
            # Infer without header; the model_fn(columns) can reorder/validate
            df = pd.read_csv(io.BytesIO(request_body), header=None)
        return df

    # JSON variants
    if ct in {"application/json"}:
        data = json.loads(request_body.decode("utf-8"))
        # Accept {"data": ...} or raw
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        # Rows as dicts -> DataFrame
        if isinstance(data, list) and (len(data) == 0 or isinstance(data[0], dict)):
            return pd.DataFrame(data)
        # 2D array -> ndarray -> we’ll wrap to DataFrame in predict_fn if columns are known
        arr = np.array(data)
        return arr

    # Fallback: try JSON
    try:
        data = json.loads(request_body.decode("utf-8"))
        if isinstance(data, list) and (len(data) == 0 or isinstance(data[0], dict)):
            return pd.DataFrame(data)
        return np.array(data)
    except Exception as e:
        raise ValueError(f"Unsupported ContentType: {content_type}") from e


def output_fn(prediction: Any, accept: Optional[str]) -> Tuple[Union[str, bytes], str]:
    """
    Serialize predictions to the requested Accept.
    """
    accept = (accept or "application/x-parquet").split(";")[0].strip().lower()

    # Parquet response (nice for batch/analytics)
    if accept == "application/x-parquet":
        return _parquet_response_from_array_or_df(prediction)

    # Default JSON
    return _to_json_payload(prediction)


# ---------- NEW: Flask glue for custom container ----------

MODEL_STATE = None  # lazy-loaded global


def get_model_state():
    global MODEL_STATE
    if MODEL_STATE is None:
        # In SageMaker, model artifacts are extracted to /opt/ml/model
        LOGGER.info("Loading model state from /opt/ml/model")
        MODEL_STATE = model_fn("/opt/ml/model")
        LOGGER.info("Model state loaded")
    return MODEL_STATE


@app.route("/ping", methods=["GET"])
def ping():
    """
    Health check endpoint for SageMaker.
    Returns 200 if model loads successfully.
    """
    try:
        _ = get_model_state()
        return "", 200
    except Exception as e:
        LOGGER.exception("Ping failed: %s", e)
        return "Model load failed", 500


@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Inference endpoint.
    Uses existing input_fn / predict_fn / output_fn.
    """
    content_type = request.headers.get("Content-Type", "application/x-parquet")
    accept = request.headers.get("Accept", "application/x-parquet")
    body = request.get_data()

    try:
        state = get_model_state()
        data = input_fn(body, content_type)
        preds = predict_fn(data, state)
        payload, out_ct = output_fn(preds, accept)

        # Flask expects (response, status, headers)
        if isinstance(payload, bytes):
            return payload, 200, {"Content-Type": out_ct}
        else:
            return payload, 200, {"Content-Type": out_ct}
    except Exception as e:
        LOGGER.exception("Invocation failed: %s", e)
        return f"Error during prediction: {e}", 500
    

# NEW: start Flask app when run as a script
if __name__ == "__main__":
    LOGGER.info("Starting Flask app on 0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080)

"""Batch inference utilities for SageMaker Pipelines.

This script downloads parquet datasets from S3, scores them with the
trained bid predictor pipeline, and writes the predictions back to S3 as
parquet.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Iterator, Tuple

import boto3
import joblib
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an S3 URI (s3://...), got: {uri}")
    bucket_key = uri[5:]
    bucket, _, key = bucket_key.partition("/")
    if not bucket or not key:
        raise ValueError(f"S3 URI must include bucket and key: {uri}")
    return bucket, key


def _download_inputs(input_uri: str, destination: Path) -> Iterator[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    bucket, prefix = _parse_s3_uri(input_uri)

    paginator = s3.get_paginator("list_objects_v2")
    has_files = False
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            local_path = destination / Path(key).name
            LOGGER.info("Downloading %s to %s", key, local_path)
            s3.download_file(bucket, key, str(local_path))
            has_files = True
            yield local_path

    if not has_files:
        # The URI may point to a single object rather than a prefix. Attempt to fetch directly.
        local_path = destination / Path(prefix).name
        LOGGER.info("Attempting direct download of %s", prefix)
        s3.download_file(bucket, prefix, str(local_path))
        if not local_path.exists():
            raise FileNotFoundError(f"No parquet objects found at {input_uri}")
        yield local_path


def _load_pipeline(model_dir: Path, model_filename: str = "pipeline.joblib"):
    model_path = model_dir / model_filename
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained pipeline not found at {model_path}. Ensure the training job saved the artifact."
        )
    LOGGER.info("Loading pipeline from %s", model_path)
    return joblib.load(model_path)


def _score_file(
    pipeline,
    parquet_path: Path,
    prediction_column: str,
    probability_column: str,
    include_features: bool,
) -> pd.DataFrame:
    LOGGER.info("Scoring parquet file: %s", parquet_path)
    frame = pd.read_parquet(parquet_path)
    LOGGER.debug("Input frame shape: %s", frame.shape)

    predictions = pipeline.predict(frame)
    result = pd.DataFrame(index=frame.index)
    if include_features:
        result = frame.copy()
    result[prediction_column] = predictions

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(frame)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            positive_class = probabilities[:, 1]
        else:
            positive_class = probabilities.ravel()
        result[probability_column] = positive_class
    return result


def _upload_outputs(local_files: Iterable[Path], output_uri: str) -> None:
    bucket, key = _parse_s3_uri(output_uri)
    s3 = boto3.client("s3")
    local_files = list(local_files)
    if not local_files:
        LOGGER.warning("No output files generated; skipping upload.")
        return

    if key.endswith(".parquet") and len(local_files) == 1:
        destination_key = key
        LOGGER.info(
            "Uploading single parquet to s3://%s/%s", bucket, destination_key
        )
        s3.upload_file(str(local_files[0]), bucket, destination_key)
        return

    prefix = key.rstrip("/") + "/"
    for file_path in local_files:
        destination_key = prefix + file_path.name
        LOGGER.info("Uploading %s to s3://%s/%s", file_path, bucket, destination_key)
        s3.upload_file(str(file_path), bucket, destination_key)


def run_batch_inference(
    model_dir: Path,
    input_s3_uri: str,
    output_s3_uri: str,
    prediction_column: str,
    probability_column: str,
    include_features: bool,
    output_local_dir: Path | None = None,
) -> None:
    pipeline = _load_pipeline(model_dir)

    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_dir = tmpdir_path / "input"
        output_dir = tmpdir_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files: list[Path] = []
        for idx, parquet_file in enumerate(_download_inputs(input_s3_uri, input_dir)):
            scored = _score_file(
                pipeline,
                parquet_file,
                prediction_column=prediction_column,
                probability_column=probability_column,
                include_features=include_features,
            )
            output_file = output_dir / f"part-{idx:05d}.parquet"
            LOGGER.info("Writing predictions to %s", output_file)
            scored.to_parquet(output_file, index=False)
            generated_files.append(output_file)

        _upload_outputs(generated_files, output_s3_uri)

        if output_local_dir is not None:
            output_local_dir.mkdir(parents=True, exist_ok=True)
            for file_path in generated_files:
                shutil.copy2(file_path, output_local_dir / file_path.name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch inference on parquet inputs.")
    parser.add_argument("--input-s3-uri", required=True, help="S3 URI of the parquet dataset to score.")
    parser.add_argument(
        "--output-s3-uri",
        required=True,
        help="Destination S3 URI for the parquet predictions.",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        help="Directory containing the trained pipeline artifact.",
    )
    parser.add_argument(
        "--prediction-column",
        default="prediction",
        help="Name of the prediction column to add to the output parquet.",
    )
    parser.add_argument(
        "--probability-column",
        default="acceptance_probability",
        help="Name of the probability column to add when predict_proba is available.",
    )
    parser.add_argument(
        "--drop-features",
        action="store_true",
        help="If set, the output parquet will only include prediction columns (no original features).",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Python logging level (e.g., INFO, DEBUG).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()))

    include_features = not args.drop_features
    output_local_dir = None
    output_env_dir = os.environ.get("SM_OUTPUT_DATA_DIR")
    if output_env_dir:
        output_local_dir = Path(output_env_dir)

    run_batch_inference(
        model_dir=Path(args.model_dir),
        input_s3_uri=args.input_s3_uri,
        output_s3_uri=args.output_s3_uri,
        prediction_column=args.prediction_column,
        probability_column=args.probability_column,
        include_features=include_features,
        output_local_dir=output_local_dir,
    )


if __name__ == "__main__":
    main()

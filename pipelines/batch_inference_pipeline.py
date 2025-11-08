"""SageMaker Pipeline for batch inference with the CatBoost bid predictor.

This module exposes a helper for constructing an AWS SageMaker Pipeline that loads a
CatBoost classifier wrapped inside a scikit-learn pipeline from the SageMaker hosted
MLflow tracking server and then runs a batch transform job.  The pipeline keeps the
SageMaker resources parameterised so callers can re-use the same definition across
multiple environments (development, staging, production).

Typical usage::

    pipeline = create_batch_inference_pipeline(
        region_name="us-west-2",
        role_arn="arn:aws:iam::123456789012:role/service-role/SageMakerRole",
        pipeline_name="bid-predictor-batch-inference",
        mlflow_tracking_uri="https://mlflow.<account>.sagemaker.aws/api/2.0/mlflow",
        mlflow_model_name="bid-predictor-catboost",
        mlflow_model_stage="Production",
        inference_image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/bid-predictor:latest",
        default_batch_input="s3://my-bucket/batch/input/",
        default_batch_output="s3://my-bucket/batch/output/",
    )
    pipeline.upsert(role_arn="arn:aws:iam::123456789012:role/service-role/SageMakerRole")
    execution = pipeline.start()

The returned pipeline contains two steps:

* **CreateCatBoostModel** – resolves the MLflow registered model version and creates a
  SageMaker model entity using the provided inference container and entry-point script.
* **CatBoostBatchTransform** – launches a SageMaker Batch Transform job that scores the
  provided dataset using the model created in the previous step.

The module does not assume how the CatBoost + scikit-learn pipeline was trained.  It only
requires that the model was logged to MLflow using the ``pyfunc`` flavour (as produced by
``mlflow.sklearn.log_model`` or the custom integration in this repository) and that the
model version has been promoted to a stage or version number accessible via the SageMaker
hosted MLflow tracking server.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import boto3
from mlflow.tracking import MlflowClient
from sagemaker.model import Model
from sagemaker.session import Session
from sagemaker.transformer import Transformer
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import TransformStep


def _get_model_artifact_uri(
    *,
    tracking_uri: str,
    model_name: str,
    model_version: Optional[str],
    model_stage: Optional[str],
) -> str:
    """Resolve the S3 URI that contains the MLflow model artifacts.

    Parameters
    ----------
    tracking_uri:
        The MLflow tracking server URI hosted on SageMaker.
    model_name:
        Name of the registered model in MLflow.
    model_version:
        Specific model version to deploy.  If omitted the latest model in ``model_stage``
        will be used instead.
    model_stage:
        MLflow model stage to resolve when ``model_version`` is not provided.  The value
        should be one of ``None``, ``"Production"``, ``"Staging"`` or any custom stage
        configured in MLflow.

    Returns
    -------
    str
        The S3 URI where MLflow stored the model artifacts.  This URI can be used as the
        ``model_data`` argument when creating a SageMaker model.
    """

    client = MlflowClient(tracking_uri=tracking_uri)

    if model_version:
        version_info = client.get_model_version(name=model_name, version=model_version)
    else:
        if not model_stage:
            raise ValueError("model_stage must be provided when model_version is not set")
        latest_versions = client.get_latest_versions(name=model_name, stages=[model_stage])
        if not latest_versions:
            raise ValueError(
                f"No versions found for model '{model_name}' in stage '{model_stage}'."
            )
        version_info = latest_versions[0]

    return client.get_model_version_download_uri(model_name, version_info.version)


def _build_pipeline_session(region_name: str) -> PipelineSession:
    """Create a pipeline-aware SageMaker session for the target AWS region."""

    boto_session = boto3.Session(region_name=region_name)
    sagemaker_session = Session(boto_session=boto_session)
    return PipelineSession(boto_session=boto_session, sagemaker_session=sagemaker_session)


def create_batch_inference_pipeline(
    *,
    region_name: str,
    role_arn: str,
    pipeline_name: str,
    mlflow_tracking_uri: str,
    mlflow_model_name: str,
    inference_image_uri: str,
    default_batch_input: str,
    default_batch_output: str,
    mlflow_model_version: Optional[str] = None,
    mlflow_model_stage: Optional[str] = "Production",
    model_entry_point: Optional[str] = None,
    model_source_dir: Optional[str] = None,
    model_environment: Optional[Dict[str, str]] = None,
    transform_content_type: str = "text/csv",
    transform_output_accept: Optional[str] = "text/csv",
    default_instance_type: str = "ml.m5.xlarge",
    default_instance_count: int = 1,
) -> Pipeline:
    """Construct a SageMaker Pipeline that performs batch inference using CatBoost.

    Parameters
    ----------
    region_name:
        AWS region where the SageMaker resources should be created.
    role_arn:
        Execution role with permissions to read the MLflow artifacts, create SageMaker
        models and run batch transform jobs.
    pipeline_name:
        Name to assign to the SageMaker pipeline.
    mlflow_tracking_uri:
        URI of the SageMaker-hosted MLflow tracking server (e.g.
        ``https://<dns-name>/api/2.0/mlflow``).
    mlflow_model_name:
        Name of the registered MLflow model that encapsulates the CatBoost classifier
        wrapped inside a scikit-learn pipeline.
    inference_image_uri:
        ECR image URI that contains the inference environment.  The image must be able to
        load the MLflow ``pyfunc`` model (for example by running ``mlflow models serve`` or
        by calling the custom inference script bundled with this repository).
    default_batch_input:
        Default S3 URI where the pipeline should look for the input dataset when executing
        the batch transform job.
    default_batch_output:
        Default S3 prefix where the batch transform predictions will be stored.
    mlflow_model_version:
        Optional explicit MLflow model version to deploy.
    mlflow_model_stage:
        Model stage to resolve when ``mlflow_model_version`` is not provided.  Defaults to
        ``"Production"``.
    model_entry_point:
        Optional path to the inference entry-point script.  When provided the file is
        uploaded alongside the model and executed by SageMaker.
    model_source_dir:
        Optional directory containing additional inference dependencies.
    model_environment:
        Environment variables to inject in the SageMaker model container.
    transform_content_type:
        MIME type for the incoming batch data.  ``text/csv`` is the default because the
        training pipeline in this project consumes CSV inputs.
    transform_output_accept:
        MIME type that the batch transform job should produce.
    default_instance_type:
        Default instance type used for the batch transform step.
    default_instance_count:
        Number of instances for the batch transform step.

    Returns
    -------
    sagemaker.workflow.pipeline.Pipeline
        Fully parameterised SageMaker pipeline ready to be submitted.
    """

    pipeline_session = _build_pipeline_session(region_name)
    model_data_uri = _get_model_artifact_uri(
        tracking_uri=mlflow_tracking_uri,
        model_name=mlflow_model_name,
        model_version=mlflow_model_version,
        model_stage=mlflow_model_stage,
    )

    model_data_param = ParameterString(name="ModelDataUri", default_value=model_data_uri)
    batch_input_param = ParameterString(name="BatchInputData", default_value=default_batch_input)
    batch_output_param = ParameterString(
        name="BatchOutputLocation", default_value=default_batch_output
    )
    instance_type_param = ParameterString(
        name="BatchInstanceType", default_value=default_instance_type
    )
    instance_count_param = ParameterInteger(
        name="BatchInstanceCount", default_value=default_instance_count
    )
    job_name_param = ParameterString(
        name="BatchTransformJobName", default_value=f"{pipeline_name}-transform"
    )

    model_kwargs = dict(
        name=pipeline_name,
        image_uri=inference_image_uri,
        model_data=model_data_param,
        role=role_arn,
        sagemaker_session=pipeline_session,
        env=model_environment or {},
    )
    if model_entry_point:
        model_kwargs["entry_point"] = model_entry_point
    if model_source_dir:
        model_kwargs["source_dir"] = model_source_dir

    model = Model(**model_kwargs)
    create_model_step = ModelStep(
        name="CreateCatBoostModel",
        step_args=model.create(),
    )

    transformer = Transformer(
        model_name=create_model_step.properties.ModelName,
        instance_type=instance_type_param,
        instance_count=instance_count_param,
        output_path=batch_output_param,
        accept=transform_output_accept,
        env=model_environment or {},
        sagemaker_session=pipeline_session,
    )

    transform_step = TransformStep(
        name="CatBoostBatchTransform",
        step_args=transformer.transform(
            data=batch_input_param,
            content_type=transform_content_type,
            job_name=job_name_param,
        ),
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            model_data_param,
            batch_input_param,
            batch_output_param,
            instance_type_param,
            instance_count_param,
            job_name_param,
        ],
        steps=[create_model_step, transform_step],
        sagemaker_session=pipeline_session,
    )

    return pipeline


def _parse_key_value_pairs(values: Optional[list[str]]) -> Dict[str, str]:
    """Parse CLI ``KEY=VALUE`` pairs into a dictionary."""

    result: Dict[str, str] = {}
    if not values:
        return result
    for pair in values:
        if "=" not in pair:
            raise ValueError(f"Invalid environment override '{pair}'. Expected KEY=VALUE format.")
        key, value = pair.split("=", 1)
        result[key] = value
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an AWS SageMaker Pipeline that performs batch inference using the "
            "CatBoost-based bid predictor registered in MLflow."
        )
    )
    parser.add_argument("--region", dest="region", required=True, help="AWS region name")
    parser.add_argument("--role-arn", dest="role_arn", required=True, help="SageMaker IAM role")
    parser.add_argument("--pipeline-name", dest="pipeline_name", required=True)
    parser.add_argument(
        "--mlflow-tracking-uri",
        dest="mlflow_tracking_uri",
        default=os.getenv("MLFLOW_TRACKING_URI"),
        help="SageMaker MLflow tracking URI (defaults to the MLFLOW_TRACKING_URI environment variable).",
    )
    parser.add_argument("--mlflow-model-name", dest="mlflow_model_name", required=True)
    parser.add_argument("--mlflow-model-version", dest="mlflow_model_version")
    parser.add_argument(
        "--mlflow-model-stage",
        dest="mlflow_model_stage",
        default="Production",
        help="MLflow model stage to resolve when a specific version is not provided.",
    )
    parser.add_argument("--inference-image-uri", dest="inference_image_uri", required=True)
    parser.add_argument("--model-entry-point", dest="model_entry_point")
    parser.add_argument("--model-source-dir", dest="model_source_dir")
    parser.add_argument(
        "--model-env",
        dest="model_env",
        action="append",
        help="Repeatable KEY=VALUE pairs injected as environment variables in the inference container.",
    )
    parser.add_argument(
        "--default-batch-input",
        dest="default_batch_input",
        required=True,
        help="Default S3 URI for the batch transform input data.",
    )
    parser.add_argument(
        "--default-batch-output",
        dest="default_batch_output",
        required=True,
        help="Default S3 prefix for the batch transform predictions.",
    )
    parser.add_argument(
        "--transform-content-type",
        dest="transform_content_type",
        default="text/csv",
    )
    parser.add_argument(
        "--transform-output-accept",
        dest="transform_output_accept",
        default="text/csv",
    )
    parser.add_argument(
        "--instance-type",
        dest="instance_type",
        default="ml.m5.xlarge",
        help="Default instance type for batch transform runs.",
    )
    parser.add_argument(
        "--instance-count",
        dest="instance_count",
        type=int,
        default=1,
        help="Default instance count for batch transform runs.",
    )
    return parser


def main(args: Optional[list[str]] = None) -> None:
    """Entry point for the command line interface."""

    parser = _build_arg_parser()
    parsed = parser.parse_args(args=args)

    if not parsed.mlflow_tracking_uri:
        parser.error(
            "MLflow tracking URI must be provided either via --mlflow-tracking-uri or the "
            "MLFLOW_TRACKING_URI environment variable."
        )

    pipeline = create_batch_inference_pipeline(
        region_name=parsed.region,
        role_arn=parsed.role_arn,
        pipeline_name=parsed.pipeline_name,
        mlflow_tracking_uri=parsed.mlflow_tracking_uri,
        mlflow_model_name=parsed.mlflow_model_name,
        mlflow_model_version=parsed.mlflow_model_version,
        mlflow_model_stage=parsed.mlflow_model_stage,
        inference_image_uri=parsed.inference_image_uri,
        default_batch_input=parsed.default_batch_input,
        default_batch_output=parsed.default_batch_output,
        model_entry_point=parsed.model_entry_point,
        model_source_dir=parsed.model_source_dir,
        model_environment=_parse_key_value_pairs(parsed.model_env),
        transform_content_type=parsed.transform_content_type,
        transform_output_accept=parsed.transform_output_accept,
        default_instance_type=parsed.instance_type,
        default_instance_count=parsed.instance_count,
    )

    pipeline_definition = pipeline.definition()
    print(pipeline_definition)


if __name__ == "__main__":
    main()

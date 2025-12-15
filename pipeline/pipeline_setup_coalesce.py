import boto3
import sagemaker
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker import get_execution_role
from sagemaker.session import Session
from sagemaker.network import NetworkConfig


ENVIRONMENT = "preprd"
if ENVIRONMENT.lower() == "dev":
    ACCOUNT_ID = "622055002283"
    REGION = "us-east-1"
    IMAGE_NAME = "bsp-data-fetching-image"

    PIPELINE_NAME = "CoalesceResultsPipelineDev"

    BUCKET_NAME = "amazon-sagemaker-622055002283-us-east-1-b37b41a56cd8"
    PREFIX = "dzd_4dt0rvdnr1hoiv/dfbsxtgjets9wn"
    DEFAULT_HWM_PATH = (
        f"s3://{BUCKET_NAME}/{PREFIX}/output"
        f"/hwm/probability_hwm/probability.txt"
    )
    DEFAULT_LOOKUP_PATH = (
        f"s3://{BUCKET_NAME}/output/lookup_file"
        f"/lookup_table_bid_predictor.csv"
    )
    DEFAULT_OUTPUT_CSV = (
        f"s3://{BUCKET_NAME}/{PREFIX}/shared"
        f"/offer_probability_csv/offer_probability.csv"
    )

    network_config = None

elif ENVIRONMENT.lower() == "stg":
    ACCOUNT_ID = "622055002283"
    REGION = "us-east-1"
    IMAGE_NAME = "bsp-data-fetching-image"

    PIPELINE_NAME = "CoalesceResultsPipelineStg"

    BUCKET_NAME = "amazon-sagemaker-622055002283-us-east-1-b37b41a56cd8"
    PREFIX = "dzd_4dt0rvdnr1hoiv/dfbsxtgjets9wn"
    DEFAULT_HWM_PATH = (
        f"s3://{BUCKET_NAME}/{PREFIX}/output"
        f"/hwm/probability_hwm/probability.txt"
    )
    DEFAULT_LOOKUP_PATH = (
        f"s3://{BUCKET_NAME}/output/lookup_file"
        f"/lookup_table_bid_predictor.csv"
    )
    DEFAULT_OUTPUT_CSV = (
        f"s3://{BUCKET_NAME}/{PREFIX}/shared"
        f"/offer_probability_csv/offer_probability.csv"
    )

    network_config = None

elif ENVIRONMENT.lower() in ("preprd", "preprod"):
    ACCOUNT_ID = "382704342560"
    REGION = "us-east-1"

    PIPELINE_NAME = "CoalesceResultsPipelinePrePrd"

    IMAGE_NAME = "bsp-data-fetching-image"

    BUCKET_NAME = "sagemaker-us-east-1-382704342560"

    DEFAULT_HWM_PATH = (
        f"s3://{BUCKET_NAME}/bid_success_predictor/"
        f"hwm/probability_hwm/probability.txt"
    )
    DEFAULT_LOOKUP_PATH = (
        f"s3://{BUCKET_NAME}/bid_success_predictor/"
        f"lookup_table_bid_predictor.csv"
    )
    DEFAULT_OUTPUT_CSV = (
        f"s3://{BUCKET_NAME}/bid_success_predictor_prd"
        f"/offer_probability_csv/offer_probability.csv"
    )
    # Create the network config
    network_config = NetworkConfig(
        subnets=['subnet-07cfb1a5945594ed2', 'subnet-094594e0b85e77bb6'],
        security_group_ids=['sg-0d755d27b93bae7cf']
    )

elif ENVIRONMENT.lower() in ("prd", "prod"):
    ACCOUNT_ID = "382704342560"
    REGION = "us-east-1"

    PIPELINE_NAME = "CoalesceResultsPipelinePrd"

    IMAGE_NAME = "bsp-data-fetching-image"

    BUCKET_NAME = "sagemaker-us-east-1-382704342560"

    DEFAULT_HWM_PATH = (
        f"s3://{BUCKET_NAME}/bid_success_predictor/"
        f"hwm/probability_hwm/probability.txt"
    )
    DEFAULT_LOOKUP_PATH = (
        f"s3://{BUCKET_NAME}/bid_success_predictor/"
        f"lookup_table_bid_predictor.csv"
    )
    DEFAULT_OUTPUT_CSV = (
        f"s3://{BUCKET_NAME}/bid_success_predictor_prd"
        f"/offer_probability_csv/offer_probability.csv"
    )

    # Create the network config
    network_config = NetworkConfig(
        subnets=['subnet-07cfb1a5945594ed2', 'subnet-094594e0b85e77bb6'],
        security_group_ids=['sg-0d755d27b93bae7cf']
    )

# script_name = "probability-py.py"
# s3_prefix = "dzd_4dt0rvdnr1hoiv/dfbsxtgjets9wn/output/scripts"
# s3_key = f"{s3_prefix}/{script_name}"

# with open(script_name, "rb") as f:
#     boto3.client("s3").put_object(Bucket=bucket, Key=s3_key, Body=f.read())

# script_s3_uri = f"s3://{bucket}/{s3_key}"
# print("Uploaded script to:", script_s3_uri)


def main():
    s3_bucket_param = ParameterString(name="s3_bucket", default_value=BUCKET_NAME)
    hwm_path_param = ParameterString(name="hwm_path", default_value=DEFAULT_HWM_PATH)
    lookup_path_param = ParameterString(name="lookup_path", default_value=DEFAULT_LOOKUP_PATH)
    output_csv_param = ParameterString(name="output_csv", default_value=DEFAULT_OUTPUT_CSV)

    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    processor = ScriptProcessor(
        role=role,
        image_uri=f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{IMAGE_NAME}:latest",
        command=["python3"],
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=30,
        max_runtime_in_seconds=86400,  # 24 hours
        base_job_name="BSPResultsCoalesce",
        sagemaker_session=sagemaker_session,
        network_config=network_config,
    )

    processing_step = ProcessingStep(
        name="BSPResultsCoalesceProcessing",
        processor=processor,
        code="coalesce_results.py",
        job_arguments=[
            "--s3_bucket", s3_bucket_param,
            "--hwm_path", hwm_path_param,
            "--lookup_path", lookup_path_param,
            "--output_csv", output_csv_param,
        ],
    )

    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            s3_bucket_param,
            hwm_path_param,
            lookup_path_param,
            output_csv_param,
        ],
        steps=[processing_step],
        sagemaker_session=sagemaker_session
    )

    pipeline.upsert(role_arn=role)


if __name__ == "__main__":
    main()
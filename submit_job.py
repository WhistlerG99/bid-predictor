# submit_job_gpu.py
import os
import pandas as pd
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from dotenv import load_dotenv

load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")
# ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
sess = sagemaker.Session()

account = sess.boto_session.client("sts").get_caller_identity()["Account"]
repo = "bid-predictor-training"
tag = "latest"
image_uri = f"{account}.dkr.ecr.{REGION}.amazonaws.com/{repo}:{tag}"

# task_type = "CPU"
# instance_type = "ml.m5.xlarge"
# devices = "0"

task_type = "GPU"
instance_type = "ml.g5.xlarge"
devices = "0"

# instance_type = "ml.g5.12xlarge" # 4 GPUs
# devices = "-1"
# devices = "0,1,2,3"

experiment_name = "bid-predictor-test-2-ey"

job_timestamp = f"{pd.Timestamp.now():%Y-%m-%d-%H-%M-%S}" #dt.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
job_name = f"{experiment_name}-{job_timestamp}"

run_name = f"run-{job_timestamp}"

# feature_config = "feature_config/feature_config_bid_rank_2_v5.yaml"
feature_config = "feature_config/feature_config_etihad.yaml"
iterations = 500

train_s3 = os.environ.get("S3_BUCKET_DATA") + "/data"

# train_s3 += "/air_canada_and_lot/bid_data_snapshots_v2.parquet"
train_s3 += "/etihad/bid_and_flight_data_snapshots_20260107.parquet"
# train_s3 += "/saudia/bid_and_flight_data_snapshots_20260107.parquet"


est = Estimator(
    image_uri=image_uri,
    # role=ROLE_ARN,
    instance_count=1,
    instance_type=instance_type,  # GPU instance
    sagemaker_session=sess,
    # base_job_name=experiment_name,
    # pass anything your train.py parses; ensure CatBoost runs on GPU
    hyperparameters={
        # only matters if your build_pipeline uses these
        "task-type": task_type,
        "devices": devices,
        "iterations": iterations,
        "experiment-name": experiment_name,
        "run-name": run_name,
        "feature-config": feature_config,
        "training-data-path": train_s3,
        "test-fraction": 0.2,
        # "scale-pos-weight": 0.25,
        "auto-class-weights": "Balanced",
        # "travel-date-min": None,
        # "travel-date-max": None,
    },
    # keep this so you can iterate code without rebuilding the image
    entry_point="train.py",
    source_dir=".",
    # (Optional) larger volume if needed for big wheels/artifacts
    # volume_size=100,
    # (Optional) env vars (MLflow SigV4, tracking URL, etc.)
    environment={
        # if you set these in code you can skip here
        # "MLFLOW_TRACKING_URI": "https://<your-tracking-url>",
        # "MLFLOW_TRACKING_REQUEST_HEADER_PROVIDER":
        #   "mlflow_sagemaker.request_header_provider:SigV4RequestHeaderProvider",
        "AWS_REGION": REGION
    },
)


inputs = {
    "train": TrainingInput(
        s3_data=train_s3,
        content_type="application/x-parquet",  # "text/csv",
        # distribution="ShardedByS3Key",
        s3_data_type="S3Prefix",
        input_mode="File",
    )
}

print(f"Submitting job: {job_name}")
est.fit(inputs, job_name=job_name, wait=True, logs=True)

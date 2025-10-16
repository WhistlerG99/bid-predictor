# submit_job_gpu.py
import os
import datetime as dt
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from dotenv import load_dotenv
load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")
# ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
sess = sagemaker.Session()

account = sess.boto_session.client("sts").get_caller_identity()["Account"]
repo = "bid-predictor-sklearn-gpu"
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

est = Estimator(
    image_uri=image_uri,
    # role=ROLE_ARN,
    instance_count=1,
    instance_type=instance_type,    # GPU instance
    sagemaker_session=sess,
    base_job_name="snapshot-bid-predictor-gpu",
    # pass anything your train.py parses; ensure CatBoost runs on GPU
    hyperparameters={
        # only matters if your build_pipeline uses these
        "task_type": task_type,
        "devices": devices
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
    }
)


train_s3 = os.environ.get("S3_BUCKET_DATA") + "/data/air_canada_and_lot/bid_data_snapshots_v2.parquet"
inputs = {
    "train": TrainingInput(
        s3_data=train_s3,
        content_type="application/x-parquet",#"text/csv",
        # distribution="ShardedByS3Key",
        s3_data_type="S3Prefix",
        input_mode="File",
    )
}

job_name = "snapshot-bid-predictor-gpu-" + dt.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
print(f"Submitting job: {job_name}")
est.fit(inputs, job_name=job_name, wait=True, logs=True)

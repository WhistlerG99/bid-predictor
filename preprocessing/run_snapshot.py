from sagemaker.spark.processing import PySparkProcessor
from sagemaker import get_execution_role

role = get_execution_role()
processor = PySparkProcessor(
    base_job_name="spark-job",
    framework_version="3.3",
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=2
)

S3_BUCKET_DATA="s3://amazon-sagemaker-622055002283-us-east-1-b37b41a56cd8/dzd_4dt0rvdnr1hoiv/5vt5uv9jpcqmxz"
DATA_PATH=f"{S3_BUCKET_DATA}/data/etihad/bid_and_flight_data_etihad.parquet"
OUT_PATH=f"{S3_BUCKET_DATA}/data/etihad/bid_and_flight_data_snapshots_etihad.parquet"

processor.run(
    submit_app=f"{S3_BUCKET_DATA}/shared/bid-predictor/preprocessing/create_auction_snapshots_spark.py",
    arguments=["--input", DATA_PATH, "--output", OUT_PATH],
)
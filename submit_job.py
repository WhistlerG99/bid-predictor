# submit_job_custom_image.py
import os, datetime as dt, sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

REGION = os.environ.get("AWS_REGION", "us-east-1")
# ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]  # same role you were using before
sess = sagemaker.Session()

account = sess.boto_session.client("sts").get_caller_identity()["Account"]
image_uri = f"{account}.dkr.ecr.{REGION}.amazonaws.com/bid-predictor-sklearn:latest"

estimator = Estimator(
    image_uri=image_uri,
    # role=ROLE_ARN,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    sagemaker_session=sess,
    base_job_name="catboost-bid-predictor",
    # you *can* still override hyperparameters if your train.py parses them
    hyperparameters={},
    entry_point="train.py",    # <— yes: you can still pass your code in source_dir
    source_dir=".",            # so you don't have to rebuild the image on code changes
)

train_s3 = "s3://amazon-sagemaker-622055002283-us-east-1-b37b41a56cd8/dzd_4dt0rvdnr1hoiv/5vt5uv9jpcqmxz/shared/bid-predictor/bid_data_enriched_new_reduced.csv"  # ends with a slash; can hold multiple files
inputs = {
    "train": TrainingInput(
        s3_data=train_s3,
        content_type="text/csv",
        input_mode="File"
    )
}

job_name = "catboost-bid-predictor-" + dt.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
print(f"Submitting job: {job_name}")
estimator.fit(inputs, job_name=job_name, wait=True, logs=True)

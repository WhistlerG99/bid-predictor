import os
from pathlib import Path


def in_sagemaker() -> bool:
    env = os.environ
    return Path("/opt/ml").exists() or any(  # any SageMaker job container
        k in env
        for k in (
            # Studio / Notebook
            "SAGEMAKER_DOMAIN_ID",
            "SAGEMAKER_USER_PROFILE_NAME",
            "SAGEMAKER_NOTEBOOK_INSTANCE_NAME",
            "SAGEMAKER_NOTEBOOK_ARN",
            # Common job signals
            "SM_TRAINING_ENV",
            "TRAINING_JOB_NAME",
            "SM_HOSTS",
            "PROCESSING_JOB_NAME",
            "SM_PROCESSING_ENV",
            "SAGEMAKER_INFERENCE",
        )
    )


def get_output_dir():
    # Prefer CATBOOST_TRAIN_DIR if present, else /opt/ml/output
    output_dir = os.environ.get("CATBOOST_TRAIN_DIR", "/opt/ml/output/catboost")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError:
        # Fall back to /tmp to avoid failing the whole job
        get_output_dir = os.path.join("/tmp", "catboost")
        os.makedirs(output_dir, exist_ok=True)
        print(f"WARNING: /opt/ml/output not writable; using {output_dir}")
    return output_dir

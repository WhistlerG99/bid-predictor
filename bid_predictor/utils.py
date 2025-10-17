import os
import os
import sys
from typing import Literal, Tuple
from pathlib import Path

EnvKind = Literal["sagemaker_job", "sagemaker_notebook", "sagemaker_terminal", "local"]


def _in_jupyter() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        return get_ipython() is not None
    except Exception:
        return False


def _has_any_env(keys) -> bool:
    env = os.environ
    return any(k in env and str(env[k]).strip() != "" for k in keys)


def _any_file_exists(paths) -> bool:
    return any(os.path.exists(p) for p in paths)


def detect_execution_environment() -> Tuple[EnvKind, str]:
    """
    Detects whether we're running in:
      - SageMaker job container ("sagemaker_job")
      - SageMaker Jupyter kernel ("sagemaker_notebook")
      - SageMaker terminal / shell ("sagemaker_terminal")
      - Local environment ("local")

    Returns:
        (env_kind, reason)
    """

    # --- 1) SageMaker *job* container (training/processing/inference) ---
    job_env_keys = [
        "SM_TRAINING_ENV",
        "SM_JOB_NAME",
        "SM_CURRENT_HOST",
        "SM_RESOURCE_CONFIG",
        "SM_INPUT_DATA_CONFIG",
    ]
    job_config_files = [
        "/opt/ml/input/config/hyperparameters.json",
        "/opt/ml/input/config/resourceconfig.json",
        "/opt/ml/input/config/inputdataconfig.json",
        "/opt/ml/config/processingjobconfig.json",
        "/opt/ml/model/.sagemaker-inference",
    ]
    if _has_any_env(job_env_keys) or _any_file_exists(job_config_files):
        return (
            "sagemaker_job",
            "Found SageMaker job config (SM_* env or /opt/ml/*config*.json).",
        )

    # Common notebook/studio signals (terminals in Studio share these env vars too)
    notebook_env_keys = [
        "SAGEMAKER_DOMAIN_ID",
        "SAGEMAKER_STUDIO_USER_PROFILE_NAME",
        "SAGEMAKER_INTERNAL_IMAGE_URI",
        "SAGEMAKER_JUPYTERSERVER_IMAGE_URI",
        "JUPYTER_SERVER_NAME",
        "SAGEMAKER_PROJECT_NAME",
    ]
    aws_exec_env = os.environ.get("AWS_EXECUTION_ENV", "")

    # --- 2) SageMaker *notebook* (Jupyter kernel) ---
    if _in_jupyter() and (
        _has_any_env(notebook_env_keys) or "AmazonSageMakerNotebook" in aws_exec_env
    ):
        return (
            "sagemaker_notebook",
            "Running in a Jupyter kernel with SageMaker Studio/Notebook indicators.",
        )

    # Legacy NI notebook heuristic: Jupyter + ~/SageMaker on Linux
    home = os.path.expanduser("~")
    if (
        _in_jupyter()
        and sys.platform.startswith("linux")
        and os.path.exists(os.path.join(home, "SageMaker"))
    ):
        return (
            "sagemaker_notebook",
            "Jupyter on Linux with ~/SageMaker suggests a SageMaker Notebook Instance.",
        )

    # --- 3) SageMaker *terminal* (shell, not Jupyter) ---
    # Studio/NI terminals typically have SAGEMAKER_* vars (Studio) or EC2-style home (NI),
    # but there is NO Jupyter IPython kernel active.
    if (not _in_jupyter()) and (
        _has_any_env(notebook_env_keys)
        or "AmazonSageMakerNotebook" in aws_exec_env
        or (
            sys.platform.startswith("linux")
            and os.path.exists(os.path.join(home, "SageMaker"))
        )
    ):
        return (
            "sagemaker_terminal",
            "No Jupyter kernel, but SageMaker Studio/Notebook environment markers present (terminal session).",
        )

    # --- 4) Otherwise, local ---
    return ("local", "No SageMaker job/notebook/terminal markers detected.")


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

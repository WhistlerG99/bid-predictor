"""Pipeline definitions for deploying bid predictor models on SageMaker."""

from .batch_inference_pipeline import create_batch_inference_pipeline

__all__ = ["create_batch_inference_pipeline"]

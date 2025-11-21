import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline

from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model

from sagemaker.transformer import Transformer
from sagemaker.workflow.steps import TransformStep
from sagemaker.inputs import TransformInput, BatchDataCaptureConfig  # CHANGED

from sagemaker.workflow.parameters import ParameterString
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep

from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join

DEV=False
if DEV:
    PIPELINE_NAME = "BidPredictorBatchInferenceDev5"
    DATA_PREFIX = (
        "dzd_4dt0rvdnr1hoiv/dfbsxtgjets9wn/output/"
        "bid_predictor_live_data_by_partners/test"
    )
    PARTNER_CODE = "EY"
    # MODEL_NAME = "snapshot-bid-predictor-gpu-2025-11-20-13-41-06"
    MODEL_NAME = "bid-predictor-ey-2025-11-21-17-24-01"
    IMAGE_NAME = "bid-predictor-sklearn-inference-gpu-test"
else:
    PIPELINE_NAME = "BidPredictorBatchInference"
    DATA_PREFIX = (
        "dzd_4dt0rvdnr1hoiv/dfbsxtgjets9wn/output/"
        "bid_predictor_live_data_by_partners"
    )
    PARTNER_CODE = "EY"
    # MODEL_NAME = "snapshot-bid-predictor-gpu-2025-11-20-13-41-06"
    MODEL_NAME = "bid-predictor-ey-2025-11-21-17-24-01"
    IMAGE_NAME = "bid-predictor-sklearn-inference-gpu"    


def main():
    pipeline_session = PipelineSession()
    role = sagemaker.get_execution_role()

    bucket = "amazon-sagemaker-622055002283-us-east-1-b37b41a56cd8"

    # Model artifacts
    prefix = (
        f"dzd_4dt0rvdnr1hoiv/5vt5uv9jpcqmxz/dev/"
        f"{MODEL_NAME}/output"
    )
    model_file_name = "model"
    model_data = f"s3://{bucket}/{prefix}/{model_file_name}.tar.gz"

    instance_type = "ml.m5.xlarge"

    image_uri = (
        f"622055002283.dkr.ecr.us-east-1.amazonaws.com/"
        f"{IMAGE_NAME}:latest"
    )

    # ---------- Model step ----------
    custom_model = Model(
        model_data=model_data,
        role=role,
        image_uri=image_uri,
        sagemaker_session=pipeline_session,
    )

    model_step = ModelStep(
        name="CreateBidPredictorModel",
        step_args=custom_model.create(instance_type=instance_type),
    )

    # ---------- Base S3 layout ----------
    # <base_dir> = s3://bucket/<DATA_PREFIX>/test
    base_dir = f"s3://{bucket}/{DATA_PREFIX}"

    # final CSV:
    # <base_dir>/output/availability-offers-probability-<timestamp>.csv
    final_output_path = Join(
        on="/",
        values=[
            base_dir,
            f"output/{PARTNER_CODE}",
        ],
    )

    # processing root:
    # <base_dir>/processing/<pipeline_exec_id>/
    processing_root = Join(
        on="/",
        values=[
            base_dir,
            "processing",
            ExecutionVariables.PIPELINE_EXECUTION_ID,
        ],
    )

    # preprocessor output:
    # <base_dir>/processing/<pipeline_exec_id>/preprocessor/
    preprocessor_output_path = Join(
        on="/",
        values=[
            processing_root,
            "preprocessor",
        ],
    )

    # transformer output:
    # <base_dir>/processing/<pipeline_exec_id>/transformer/
    transformer_output_path = Join(
        on="/",
        values=[
            processing_root,
            "transformer",
        ],
    )

    # data capture path:
    # <base_dir>/processing/<pipeline_exec_id>/data_capture/
    data_capture_path = Join(  # CHANGED
        on="/",
        values=[
            processing_root,
            "data_capture",
        ],
    )

    # ---------- Input parameter ----------
    input_data_uri = ParameterString(
        name="InputData",
        default_value=f"{base_dir}/{PARTNER_CODE}/",
    )

    # ---------- Preprocessing step ----------
    script_processor = ScriptProcessor(
        role=role,
        image_uri=image_uri,
        command=["python3"],
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=pipeline_session,
    )

    # preprocess.py is expected to write /opt/ml/processing/output/processed.parquet
    processing_step = ProcessingStep(
        name="PreprocessOffers",
        processor=script_processor,
        code="preprocess.py",
        inputs=[
            ProcessingInput(
                source=input_data_uri,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="preprocessed",
                source="/opt/ml/processing/output",
                destination=preprocessor_output_path,
            )
        ],
    )

    # ---------- Batch Transform step ----------
    # Keep everything as Parquet between preprocessor and transformer
    transformer = Transformer(
        model_name=model_step.properties.ModelName,
        instance_count=1,
        instance_type=instance_type,
        output_path=transformer_output_path,
        accept="application/x-parquet",      # model returns Parquet
        sagemaker_session=pipeline_session,
    )

    batch_transform_step = TransformStep(
        name="BatchInference",
        transformer=transformer,
        inputs=TransformInput(
            data=processing_step
                 .properties
                 .ProcessingOutputConfig
                 .Outputs["preprocessed"]
                 .S3Output
                 .S3Uri,
            content_type="application/x-parquet",  # model expects Parquet
            split_type="None",
            batch_data_capture_config=BatchDataCaptureConfig(
                destination_s3_uri=data_capture_path  
            ),
        ),
    )

    # ---------- Postprocessing to CSV ----------
    # postprocess.py should:
    # - read *.parquet / *.parquet.out from /opt/ml/processing/input
    # - write availability-offers-probability-<timestamp>.csv to /opt/ml/processing/output
    postprocess_processor = ScriptProcessor(
        role=role,
        image_uri=image_uri,
        command=["python3"],
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=pipeline_session,
    )

    postprocess_step = ProcessingStep(
        name="PostprocessToCsv",
        processor=postprocess_processor,
        code="postprocess.py",
        inputs=[
            ProcessingInput(
                source=batch_transform_step
                       .properties
                       .TransformOutput
                       .S3OutputPath,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="final_csv",
                source="/opt/ml/processing/output",
                destination=final_output_path,
            )
        ],
    )

    # ---------- Pipeline ----------
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[input_data_uri],
        steps=[model_step, processing_step, batch_transform_step, postprocess_step],
        sagemaker_session=pipeline_session,
    )

    pipeline.upsert(role_arn=role)


if __name__ == "__main__":
    main()

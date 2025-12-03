import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline

from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model

from sagemaker.transformer import Transformer
from sagemaker.workflow.steps import TransformStep
from sagemaker.inputs import TransformInput, BatchDataCaptureConfig

from sagemaker.workflow.parameters import ParameterString
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep

from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.properties import PropertyFile


BUCKET_NAME = "amazon-sagemaker-622055002283-us-east-1-b37b41a56cd8"
MODEL_BASE_PREFIX = "dzd_4dt0rvdnr1hoiv/5vt5uv9jpcqmxz/dev"

DEV = True
if DEV:
    PIPELINE_NAME = "BidPredictorBatchInferenceDev5"
    DATA_PREFIX = (
        "dzd_4dt0rvdnr1hoiv/dfbsxtgjets9wn/output/"
        "bid_predictor_live_data_by_partners_test"
    )
    IMAGE_NAME = "bid-predictor-sklearn-inference-gpu-test"
    MODEL_NAME_PREFIX = "bid-predictor-test"
else:
    PIPELINE_NAME = "BidPredictorBatchInference"
    DATA_PREFIX = (
        "dzd_4dt0rvdnr1hoiv/dfbsxtgjets9wn/output/"
        "bid_predictor_live_data_by_partners"
    )
    IMAGE_NAME = "bid-predictor-sklearn-inference-gpu"
    MODEL_NAME_PREFIX = "bid-predictor"


def main():
    pipeline_session = PipelineSession()
    role = sagemaker.get_execution_role()

    instance_type = "ml.m5.xlarge"

    image_uri = (
        f"622055002283.dkr.ecr.us-east-1.amazonaws.com/"
        f"{IMAGE_NAME}:latest"
    )

    # ---------- Base S3 layout ----------
    base_dir = f"s3://{BUCKET_NAME}/{DATA_PREFIX}"

    # ---------- Input parameters ----------
    # InputPrefix represents the FULL S3 key of the input file
    # e.g. "dzd_.../output/bid_predictor_live_data_by_partners/<partner>/<file>.parquet"
    input_prefix = ParameterString(
        name="InputPrefix",
        default_value=(
            "dzd_4dt0rvdnr1hoiv/dfbsxtgjets9wn/output/"
            "bid_predictor_live_data_by_partners/EY/availability-offers-example.parquet"
        ),
    )

    # Build full S3 URI from bucket + input_prefix
    input_data_uri = Join(
        on="",  # use empty separator for proper URI assembly
        values=[
            "s3://",
            BUCKET_NAME,
            "/",
            input_prefix,
        ],
    )

    # final CSV:
    # <base_dir>/output/<partner>/availability-offers-probability-<timestamp>.csv
    final_output_path = Join(
        on="/",
        values=[
            base_dir,
            "output",
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

    # model_config output
    # <base_dir>/processing/<pipeline_exec_id>/model_config/
    model_config_output_path = Join(  # CHANGED
        on="/",
        values=[
            processing_root,
            "model_config",
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
    data_capture_path = Join(
        on="/",
        values=[
            processing_root,
            "data_capture",
        ],
    )

    # ---------- Preprocessing step ----------
    # ScriptProcessor now gets env vars so preprocess.py can select the model
    script_processor = ScriptProcessor(
        role=role,
        image_uri=image_uri,
        command=["python3"],
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=pipeline_session,
        env={  # pass model bucket & base prefix for dynamic selection
            "MODEL_BUCKET": BUCKET_NAME,
            "MODEL_BASE_PREFIX": MODEL_BASE_PREFIX,
            "MODEL_NAME_PREFIX": MODEL_NAME_PREFIX,
        },
    )

    # PropertyFile now points to output_name="model_config"
    model_config_prop = PropertyFile(
        name="ModelConfig",
        output_name="model_config",
        path="model_config.json",
    )

    # preprocess.py is expected to:
    # - write /opt/ml/processing/output/processed.parquet
    # - write /opt/ml/processing/output/model_config.json with {"model_data": "s3://.../model.tar.gz"}
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
            ),
            ProcessingOutput(  # second output for model_config
                output_name="model_config",
                source="/opt/ml/processing/model_config",
                destination=model_config_output_path,
            ),            
        ],
        property_files=[model_config_prop],  # expose model_config.json as step properties
    )

    # Use JsonGet to read "model_data" from model_config.json
    dynamic_model_data = JsonGet(
        step_name=processing_step.name,
        property_file=model_config_prop,
        json_path="model_data",
    )

    # ---------- Model step ----------
    # Use dynamic_model_data instead of hard-coded S3 URI
    custom_model = Model(
        model_data=dynamic_model_data,  # CHANGED
        role=role,
        image_uri=image_uri,
        sagemaker_session=pipeline_session,
    )

    model_step = ModelStep(
        name="CreateBidPredictorModel",
        step_args=custom_model.create(instance_type=instance_type),
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
    # Ordering so that Preprocess (which selects model) runs before ModelStep
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[input_prefix],
        steps=[
            processing_step,       # must run first to produce model_data + processed.parquet
            model_step,
            batch_transform_step,
            postprocess_step,
        ],
        sagemaker_session=pipeline_session,
    )

    pipeline.upsert(role_arn=role)


if __name__ == "__main__":
    main()

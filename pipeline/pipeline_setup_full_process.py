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

from sagemaker.network import NetworkConfig

DEFAULT_REDSHIFT_HOST_STG = (
    "redshift-serverless-wg-stg.234771642813"
    ".us-east-1.redshift-serverless.amazonaws.com"
)
DEFAULT_REDSHIFT_PWD_STG = "/+R5T11uYH,x=zB[k"

DEFAULT_REDSHIFT_HOST_PRD = (
    "redshift-serverless-wg-prd.234771642813"
    ".us-east-1.redshift-serverless.amazonaws.com"
)
DEFAULT_REDSHIFT_PWD_PRD = "5A)J6^US6HG0{q)@v;£"

DEFAULT_REDSHIFT_USER = "sagemaker"
DEFAULT_REDSHIFT_PORT = "5439"


ENVIRONMENT = "dev"
if ENVIRONMENT.lower() == "dev":
    ACCOUNT_ID = "622055002283"
    REGION = "us-east-1"

    PIPELINE_NAME = "BidPredictorBatchInferenceDev6"

    BUCKET_NAME = "amazon-sagemaker-622055002283-us-east-1-b37b41a56cd8"
    MODEL_BASE_PREFIX = "dzd_4dt0rvdnr1hoiv/5vt5uv9jpcqmxz/dev"

    DATA_PREFIX = (
        "dzd_4dt0rvdnr1hoiv/dfbsxtgjets9wn/output/"
        "bid_predictor_live_data_by_partners_test"
    )

    DEFAULT_REDSHIFT_HOST = DEFAULT_REDSHIFT_HOST_STG
    DEFAULT_REDSHIFT_PWD = DEFAULT_REDSHIFT_PWD_STG
    DEFAULT_LOOKUP =f"s3://{BUCKET_NAME}/{DATA_PREFIX}/lookup_table_bid_predictor.csv"
    DEFAULT_TEMP_PATH = f"s3://{BUCKET_NAME}/{DATA_PREFIX}/processing"

    IMAGE_NAME = "bid-predictor-inference-test"
    MODEL_NAME_PREFIX = "bid-predictor-test"

    OUTPUT_BUCKET_NAME = BUCKET_NAME
    OUTPUT_DATA_PREFIX = DATA_PREFIX+"/output"

    network_config = None

elif ENVIRONMENT.lower() == "stg":
    ACCOUNT_ID = "622055002283"
    REGION = "us-east-1"
    
    PIPELINE_NAME = "BidPredictorBatchInference"

    BUCKET_NAME = "amazon-sagemaker-622055002283-us-east-1-b37b41a56cd8"
    MODEL_BASE_PREFIX = "dzd_4dt0rvdnr1hoiv/5vt5uv9jpcqmxz/dev"

    DATA_PREFIX = (
        "dzd_4dt0rvdnr1hoiv/dfbsxtgjets9wn/output/"
        "bid_predictor_live_data_by_partners"
    )

    DEFAULT_REDSHIFT_HOST = DEFAULT_REDSHIFT_HOST_STG
    DEFAULT_REDSHIFT_PWD = DEFAULT_REDSHIFT_PWD_STG
    DEFAULT_LOOKUP =f"s3://{BUCKET_NAME}/{DATA_PREFIX}/lookup_table_bid_predictor.csv"
    DEFAULT_TEMP_PATH = f"s3://{BUCKET_NAME}/{DATA_PREFIX}/processing"

    IMAGE_NAME = "bid-predictor-inference"
    MODEL_NAME_PREFIX = "bid-predictor"

    OUTPUT_BUCKET_NAME = "ffr-bsp-model-predictions"
    OUTPUT_DATA_PREFIX = "bid_success_predictor_stg"

    network_config = None

elif ENVIRONMENT.lower() in ("preprd", "preprod"):
    ACCOUNT_ID = "382704342560"
    REGION = "us-east-1"

    PIPELINE_NAME = "BidPredictorBatchInferencePrePrd"

    BUCKET_NAME = "sagemaker-us-east-1-382704342560"
    MODEL_BASE_PREFIX = "bid_success_predictor/models"

    DATA_PREFIX = "bid_success_predictor/output/bid_predictor_live_data_by_partners"

    IMAGE_NAME = "bid-predictor-inference"
    MODEL_NAME_PREFIX = "bid-predictor"

    DEFAULT_REDSHIFT_HOST = DEFAULT_REDSHIFT_HOST_PRD
    DEFAULT_REDSHIFT_PWD = DEFAULT_REDSHIFT_PWD_PRD
    DEFAULT_LOOKUP = "s3://sagemaker-us-east-1-382704342560/bid_success_predictor/lookup_table_bid_predictor.csv"    
    DEFAULT_TEMP_PATH = "s3://sagemaker-us-east-1-382704342560/bid_success_predictor/tmp"    

    OUTPUT_BUCKET_NAME = BUCKET_NAME
    OUTPUT_DATA_PREFIX = "bid_success_predictor_prd/output/results"

    network_config = NetworkConfig(
        subnets=['subnet-07cfb1a5945594ed2', 'subnet-094594e0b85e77bb6'],
        security_group_ids=['sg-0d755d27b93bae7cf']
    )

elif ENVIRONMENT.lower() in ("prd", "prod"):
    ACCOUNT_ID = "382704342560"
    REGION = "us-east-1"

    PIPELINE_NAME = "BidPredictorBatchInferencePrd"

    BUCKET_NAME = "sagemaker-us-east-1-382704342560"
    MODEL_BASE_PREFIX = "bid_success_predictor/models"

    DATA_PREFIX = "bid_success_predictor/output/bid_predictor_live_data_by_partners"

    IMAGE_NAME = "bid-predictor-inference"
    MODEL_NAME_PREFIX = "bid-predictor"

    DEFAULT_REDSHIFT_HOST = DEFAULT_REDSHIFT_HOST_PRD
    DEFAULT_REDSHIFT_PWD = DEFAULT_REDSHIFT_PWD_PRD
    DEFAULT_LOOKUP = "s3://sagemaker-us-east-1-382704342560/bid_success_predictor/lookup_table_bid_predictor.csv"    
    DEFAULT_TEMP_PATH = "s3://sagemaker-us-east-1-382704342560/bid_success_predictor/tmp"    
    
    OUTPUT_BUCKET_NAME = BUCKET_NAME
    OUTPUT_DATA_PREFIX = "bid_success_predictor_prd/output/results"

    network_config = NetworkConfig(
        subnets=['subnet-07cfb1a5945594ed2', 'subnet-094594e0b85e77bb6'],
        security_group_ids=['sg-0d755d27b93bae7cf']
    )



def main():

    host_param = ParameterString(name="host", default_value=DEFAULT_REDSHIFT_HOST)
    database_param = ParameterString(name="database", default_value="dev")
    user_param = ParameterString(name="user", default_value=DEFAULT_REDSHIFT_USER)
    password_param = ParameterString(name="password", default_value=DEFAULT_REDSHIFT_PWD)
    port_param = ParameterString(name="port", default_value=DEFAULT_REDSHIFT_PORT)
    carrier_param = ParameterString(name="carrier", default_value="SV")
    partner_csv_s3_path_param = ParameterString(name="partner_csv_s3_path", default_value=DEFAULT_LOOKUP)
    temp_s3_path_param = ParameterString(name="temp_s3_path", default_value=DEFAULT_TEMP_PATH)
    
    pipeline_session = PipelineSession()
    role = sagemaker.get_execution_role()

    # Create the network config


    instance_type = "ml.m5.xlarge"

    image_uri = (
        f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/"
        f"{IMAGE_NAME}:latest"
    )

    # ---------- Base S3 layout ----------
    base_dir = f"s3://{BUCKET_NAME}/{DATA_PREFIX}"

    # ---------- Input parameters ----------
    # InputPrefix represents the FULL S3 key of the input file
    # e.g. "dzd_.../output/bid_predictor_live_data_by_partners/<partner>/<file>.parquet"
    # input_prefix = ParameterString(
    #     name="InputPrefix",
    # )

    # Build full S3 URI from bucket + input_prefix
    # input_data_uri = Join(
    #     on="",  # use empty separator for proper URI assembly
    #     values=[
    #         "s3://",
    #         BUCKET_NAME,
    #         "/",
    #         input_prefix,
    #     ],
    # )

    # final parquet:
    final_output_path = Join(
        on="/",
        values=[
            "s3:/",
            OUTPUT_BUCKET_NAME,
            OUTPUT_DATA_PREFIX,
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
    # ScriptProcessor now gets env vars so preprocess_full_process.py can select the model

    script_processor = ScriptProcessor(
        role=role,
        image_uri=image_uri,
        command=["python3"],
        instance_type=instance_type,
        instance_count=1,
        volume_size_in_gb=50,
        max_runtime_in_seconds=86400,  # 24 hours
        base_job_name="BSPDataFecthing",        
        sagemaker_session=pipeline_session,
        env={  # pass model bucket & base prefix for dynamic selection
            "MODEL_BUCKET": BUCKET_NAME,
            "MODEL_BASE_PREFIX": MODEL_BASE_PREFIX,
            "MODEL_NAME_PREFIX": MODEL_NAME_PREFIX,
        },
        network_config=network_config,
    )

    # PropertyFile now points to output_name="model_config"
    model_config_prop = PropertyFile(
        name="ModelConfig",
        output_name="model_config",
        path="model_config.json",
    )

    # preprocess_full_process.py is expected to:
    # - write /opt/ml/processing/output/processed.parquet
    # - write /opt/ml/processing/output/model_config.json with {"model_data": "s3://.../model.tar.gz"}
    processing_step = ProcessingStep(
        name="PreprocessOffers",
        processor=script_processor,
        code="preprocess_full_process.py",
        # inputs=[
        #     ProcessingInput(
        #         source=input_data_uri,
        #         destination="/opt/ml/processing/input",
        #     )
        # ],
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
        job_arguments=[
            "--host", host_param,
            "--database", database_param,
            "--user", user_param,
            "--password", password_param,
            "--port", port_param,
            "--carrier", carrier_param,
            "--partner_csv_s3_path", partner_csv_s3_path_param,
            "--temp_s3_path", temp_s3_path_param,
        ],
        inputs=[]
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

    # ---------- Postprocessing to Output ----------
    postprocess_processor = ScriptProcessor(
        role=role,
        image_uri=image_uri,
        command=["python3"],
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=pipeline_session,
        network_config=network_config,
    )

    postprocess_step = ProcessingStep(
        name="PostprocessToOutputResults",
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
                output_name="final_output",
                source="/opt/ml/processing/output",
                destination=final_output_path,
            )
        ],
    )

    # ---------- Pipeline ----------
    # Ordering so that Preprocess (which selects model) runs before ModelStep
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            host_param,
            database_param,
            user_param,
            password_param,
            port_param,
            carrier_param,
            partner_csv_s3_path_param,
            temp_s3_path_param,
        ],
        # parameters=[input_prefix],
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

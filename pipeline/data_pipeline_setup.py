from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.processing import ScriptProcessor
from sagemaker import image_uris
import sagemaker
from sagemaker.network import NetworkConfig

ACCOUNT_ID = "382704342560"
REGION = "us-east-1"
IMAGE_NAME = "bsp-data-fetching-image"
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
region = sagemaker_session.boto_region_name

host_param = ParameterString(name="host", default_value="redshift-serverless-wg-prd.234771642813.us-east-1.redshift-serverless.amazonaws.com")
database_param = ParameterString(name="database", default_value="dev")
user_param = ParameterString(name="user", default_value="sagemaker")
password_param = ParameterString(name="password", default_value="5A)J6^US6HG0{q)@v;£")
port_param = ParameterString(name="port", default_value="5439")
partner_csv_s3_path_param = ParameterString(name="partner_csv_s3_path", default_value="s3://sagemaker-us-east-1-382704342560/bid_success_predictor/lookup_table_bid_predictor.csv")
temp_s3_path_param = ParameterString(name="temp_s3_path", default_value="s3://sagemaker-us-east-1-382704342560/bid_success_predictor/tmp")

# sklearn_image = image_uris.retrieve(framework='sklearn', region=region, version='1.2-1')
# image_uri="622055002283.dkr.ecr.us-east-1.amazonaws.com/bsp-data-fetching-image:latest"


# Create the network config
network_config = NetworkConfig(
    subnets=['subnet-07cfb1a5945594ed2', 'subnet-094594e0b85e77bb6'],
    security_group_ids=['sg-0d755d27b93bae7cf']
)


script_processor = ScriptProcessor(
    role=role,
    image_uri=f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{IMAGE_NAME}:latest",
    command=["python3"],
    instance_count=1,
    instance_type="ml.m5.4xlarge",
    volume_size_in_gb=50,
    max_runtime_in_seconds=86400,  # 24 hours
    base_job_name="BSPDataFecthing",
    sagemaker_session=sagemaker_session,
    network_config=network_config,
)


processing_step = ProcessingStep(
    name="BSPFetchAvailabilityOffersData",
    processor=script_processor,
    code="bid_predictor_live_data_single_file.py",
    job_arguments=[
        "--host", host_param,
        "--database", database_param,
        "--user", user_param,
        "--password", password_param,
        "--port", port_param,
        "--partner_csv_s3_path", partner_csv_s3_path_param,
        "--temp_s3_path", temp_s3_path_param,
    ],
    inputs=[]
)

pipeline = Pipeline(
    name="BSPAvailabilityOffersDataPipeline",
    parameters=[
        host_param,
        database_param,
        user_param,
        password_param,
        port_param,
        partner_csv_s3_path_param,
        temp_s3_path_param,
    ],
    steps=[processing_step],
    sagemaker_session=sagemaker_session
)

pipeline.upsert(role_arn=role)
print("Pipeline created/updated successfully.")
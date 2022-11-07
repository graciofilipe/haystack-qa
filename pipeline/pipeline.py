import google.cloud.aiplatform as aip
from google_cloud_pipeline_components.experimental.custom_job import utils
from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component

from google.cloud import storage

from training_haystack.training_component import training_comp
from load_and_test.test_component import test_comp
from build_and_deploy.build_deploy_comp import deploy_comp
from build_and_deploy.build_deploy_client_comp import deploy_client_comp

#### copy artifacts to GCS ###
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

upload_blob("filipegracio-haystack", source_file_name="build_and_deploy/Dockerfile.app", destination_blob_name="pipeline/tmp/Dockerfile.app")
upload_blob("filipegracio-haystack", source_file_name="build_and_deploy/main.py", destination_blob_name="pipeline/tmp/main.py")





@dsl.pipeline(name="haystack-training")
def pipeline():
    training_op = training_comp().set_caching_options(True)


    print("hyea")

    test_op = test_comp(
        bucket_name ="filipegracio-haystack",
        artifact_path=training_op.outputs["artifact_path"],
        index_faiss_file_path=training_op.outputs["index_faiss_file_path"],
        index_json_file_path=training_op.outputs["index_json_file_path"],
        document_db_path=training_op.outputs["document_db_path"],
        pipeline_yaml_path=training_op.outputs["pipeline_yaml_path"]
    ).set_caching_options(True)

    deploy_op = deploy_client_comp(
        bucket_name ="filipegracio-haystack",
        deploy_decision=test_op.outputs["deploy_decision"],
        location_of_app_docker="/pipeline/tmp/Dockerfile.app",
        location_of_app_py="/pipeline/tmp/main.py"
        ).set_caching_options(False)

compiler = compiler.Compiler()

compiler.compile(
pipeline_func=pipeline, 
package_path="training_spec.json"
)


job = aip.PipelineJob(
    display_name="train-haystack",
    template_path="training_spec.json",
    pipeline_root="gs://filipegracio-haystack/pipeline",
    # enable_caching=True, 
    location="europe-west1"
)

job.run(service_account="257470209980-compute@developer.gserviceaccount.com")
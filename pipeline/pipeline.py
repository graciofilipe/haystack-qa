import argparse

def main(project_name,
    service_account,
        bucket_name,
        data_path,
        artifact_path,
        docker_image_name):

    import google.cloud.aiplatform as aip
    from google_cloud_pipeline_components.experimental.custom_job import utils
    from kfp.v2 import compiler, dsl
    from kfp.v2.dsl import component

    from google.cloud import storage

    from training.training_component import training_comp
    from build.build_comp_script import build_comp
    from deploy.deploy_comp_script import deploy_comp

    #### copy artifacts to GCS ###
    def upload_blob(bucket_name, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)


    upload_blob(bucket_name=bucket_name, source_file_name="build/app.Dockerfile", destination_blob_name="pipeline/tmp/app.Dockerfile")
    upload_blob(bucket_name=bucket_name, source_file_name="build/main.py", destination_blob_name="pipeline/tmp/main.py")


    @dsl.pipeline(name="haystack-training")
    def pipeline(project_name: str = None,
                bucket_name: str = None, 
                data_path: str =None,
                artifact_path: str = None,
                docker_image_name: str = None, 
                deploy_service_account: str = None):

        training_op = training_comp(bucket_name=bucket_name,
                                    data_path=data_path,
                                    artifact_path=artifact_path)\
                                        .set_caching_options(True)\
                                        .add_node_selector_constraint('cloud.google.com/gke-accelerator', 'NVIDIA_TESLA_T4')\
                                        .set_gpu_limit(1)

        build_op = build_comp(
            bucket_name =bucket_name,
            location_of_app_docker="/pipeline/tmp/app.Dockerfile",
            location_of_app_py="/pipeline/tmp/main.py", 
            docker_image_name = docker_image_name,
            training_complete = training_op.outputs["training_complete"]
            ).set_caching_options(False)

        deploy_op = deploy_comp(
            bucket_name =bucket_name,
            app_name=docker_image_name, 
            artifact_path=artifact_path,
            image_to_deploy=build_op.outputs["build_name"],
            service_account=deploy_service_account,
            ).set_caching_options(False)

    # we compile the pipeline defined above
    compiler = compiler.Compiler()
    compiler.compile(
    pipeline_func=pipeline, 
    package_path="compiled_pipeline.json"
    )

    # creat the job specs
    job = aip.PipelineJob(
        display_name="train-haystack",
        template_path="compiled_pipeline.json",
        pipeline_root="gs://" + bucket_name + "/pipeline",
        location="europe-west4",
        parameter_values={"project_name": project_name,,
                                "bucket_name": bucket_name,
                        "data_path": data_path, 
                        "artifact_path": artifact_path,
                        "docker_image_name":docker_image_name, 
                        "deploy_service_account":service_account
                        }
    )

    # run the pipeline!
    job.run(service_account=service_account)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name")
    parser.add_argument("--service_account")
    parser.add_argument("--bucket_name")
    parser.add_argument("--data_path")
    parser.add_argument("--artifact_path")
    parser.add_argument("--docker_image_name")

    args = parser.parse_args()
    main(project_name=args.project_name, 
        service_account=args.service_account,
        bucket_name=args.bucket_name,
        data_path=args.data_path,
        artifact_path=args.artifact_path,
        docker_image_name=args.docker_image_name)

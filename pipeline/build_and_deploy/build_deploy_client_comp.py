from kfp.v2.dsl import component
from typing import NamedTuple


@component(
    base_image="europe-docker.pkg.dev/filipegracio-ai-learning/haystack-docker-repo/haystack-deploy-comp:tag1",
)
def deploy_client_comp(
    bucket_name: str ="filipegracio-haystack", 
    deploy_decision: str = None,
    location_of_app_docker: str = None,
    location_of_app_py: str = None,
    ) -> NamedTuple(
    "Outputs",
    [
        ("pushed_image", str),  
    ],
):

    if deploy_decision == "true":
        import logging
        import os

        from google.cloud.devtools import cloudbuild_v1 as cloudbuild
        from google.protobuf.duration_pb2 import Duration

        # initialize client for cloud build
        logging.getLogger().setLevel(logging.INFO)
        build_client = cloudbuild.services.cloud_build.CloudBuildClient()

        build = cloudbuild.Build()
        build.steps = [

            {
                "name": "gcr.io/cloud-builders/gsutil",
                "args": [ 'cp', 'gs://' + bucket_name + location_of_app_docker, '.']
            },
                        {
                "name": "gcr.io/cloud-builders/gsutil",
                "args": [ 'cp', 'gs://' + bucket_name + location_of_app_py, '.']
            },
            {
                "name": "gcr.io/cloud-builders/docker",
                "args": [ 'build', '-t', 'europe-docker.pkg.dev/filipegracio-ai-learning/haystack-docker-repo/haystack-app:tag1', '-f' ,'Dockerfile.app', "."]
            },
            {
                "name": "gcr.io/cloud-builders/docker",
                "args": ['push', 'europe-docker.pkg.dev/filipegracio-ai-learning/haystack-docker-repo/haystack-app:tag1']
            },
        ]
        # override default timeout of 10min
        timeout = Duration()
        timeout.seconds = 7200
        build.timeout = timeout

        # create build
        operation = build_client.create_build(project_id="filipegracio-ai-learning", build=build)
        logging.info("IN PROGRESS:")
        logging.info(operation.metadata)

        # get build status
        result = operation.result()
        logging.info("RESULT:", result.status)

        # return step outputs
        print("thanks, deployed")
        return ("'europe-docker.pkg.dev/filipegracio-ai-learning/haystack-docker-repo/haystack-app:tag1'",)

    else:
        print("deploy decision was not == true")

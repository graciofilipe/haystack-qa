from kfp.v2.dsl import component
from typing import NamedTuple


@component(
    base_image="europe-docker.pkg.dev/{project_name}/haystack-docker-repo/haystack-build-comp:latest",
)
def build_comp(
    project_name: str = None,
    bucket_name: str = None, 
    location_of_app_docker: str = None,
    location_of_app_py: str = None,
    docker_image_name: str = None,
    training_complete: str = None
    ) -> NamedTuple(
    "Outputs",
    [
        ("build_name", str),  
    ],
):

    if training_complete == "true":
        import logging
        import os

        from google.cloud.devtools import cloudbuild_v1 as cloudbuild
        from google.protobuf.duration_pb2 import Duration

        # initialize client for cloud build
        logging.getLogger().setLevel(logging.INFO)
        build_client = cloudbuild.services.cloud_build.CloudBuildClient()

        build = cloudbuild.Build()
        build.steps = [
            
            #First stwo steps download the container image and the app definition into the executor. 
            # (they are uploaded when the pipeline starts)
            {
                "name": "gcr.io/cloud-builders/gsutil",
                "args": [ 'cp', 'gs://' + bucket_name + location_of_app_docker, '.']
            },
                        {
                "name": "gcr.io/cloud-builders/gsutil",
                "args": [ 'cp', 'gs://' + bucket_name + location_of_app_py, '.']
            },

            # Last two steps build the docker image here, and then push it to the artifact registry
            {
                "name": "gcr.io/cloud-builders/docker",
                "args": [ 'build', '-t', 'europe-docker.pkg.dev/' + project_name + '/haystack-docker-repo/haystack-app-' + docker_image_name +':latest', '-f' ,'app.Dockerfile', "."]
            },
            {
                "name": "gcr.io/cloud-builders/docker",
                "args": ['push', 'europe-docker.pkg.dev/' + project_name + '/haystack-docker-repo/haystack-app-' + docker_image_name +':latest']
            },
        ]
        # override default timeout of 10min
        timeout = Duration()
        timeout.seconds = 7200
        build.timeout = timeout

        # create build
        operation = build_client.create_build(project_id=project_name, build=build)
        logging.info("IN PROGRESS:")
        logging.info(operation.metadata)

        # get build status
        result = operation.result()
        logging.info("RESULT:", result.status)

        # return step outputs
        print("thanks, deployed")
        return ('europe-docker.pkg.dev/' + project_name + '/haystack-docker-repo/haystack-app-' + docker_image_name +':latest',)

    else:
        return("false",)

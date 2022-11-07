from kfp.v2.dsl import component
from typing import NamedTuple



@component(
    base_image="europe-docker.pkg.dev/filipegracio-ai-learning/haystack-docker-repo/haystack-deploy-comp:tag1",
)
def deploy_comp(
    bucket_name: str ="filipegracio-haystack",
    artifact_path: str = None,
    index_faiss_file_path: str = None,
    index_json_file_path: str = None,
    document_db_path: str = None,
    pipeline_yaml_path: str = None,
    deploy_decision: str = None,
    ):

    import subprocess
    import os
    subprocess.run(["ls", "-l"])
    subprocess.run(["pip", "install", "gcloud"])
    #os.system("gcloud builds submit --config builddeploy.app.yaml")
    subprocess.run(["gcloud", "builds", "submit", "--config", "builddeploy.app.yaml"])
 

    print("Thanks")




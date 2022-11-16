from kfp.v2.dsl import component
from typing import NamedTuple


@component(
    base_image="gcr.io/google.com/cloudsdktool/cloud-sdk"
)
def deploy_comp(
    project_name: str = None,
    bucket_name: str = None,
    app_name: str = None, 
    artifact_path: str = None,
    image_to_deploy: str = None,
    service_account: str = None,
    ) -> NamedTuple(
    "Outputs",
    [
        ("deploy_complete", str),  
    ],
):

    import subprocess

    subprocess.run( ["gcloud", "run", "deploy", app_name, "--cpu=2", "--project=" + project_name, 
                    "--min-instances=0", "--max-instances=1", "--memory=8Gi", 
                    "--port=80",  "--timeout=30m",  "--set-env-vars=artifact_path="+artifact_path+",bucket_name="+bucket_name,
                    "--image="+image_to_deploy, "--service-account="+ service_account, 
                    "--region=europe-west1", "--no-allow-unauthenticated"])
    

    return ("true",)


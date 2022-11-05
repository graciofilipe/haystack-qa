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



    # def download_files(bucket_name,
    #                 prefix,
    #                 dl_dir="./"):

    #         storage_client = storage.Client()
    #         bucket = storage_client.get_bucket(bucket_name)
    #         blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
    #         for blob in blobs:
    #             if blob.name.endswith("/"):
    #                 continue
    #             file_split = blob.name.split("/")
    #             directory = "/".join(file_split[0:-1])
    #             Path(directory).mkdir(parents=True, exist_ok=True)
    #             blob.download_to_filename(dl_dir + blob.name)


    # download_files(bucket_name=bucket_name,
    #                prefix=index_faiss_file_path,
    #                dl_dir="./")

    # download_files(bucket_name=bucket_name,
    #                prefix=index_json_file_path,
    #                dl_dir="./")

    # download_files(bucket_name=bucket_name,
    #                prefix=document_db_path,
    #                dl_dir="./")

    # download_files(bucket_name=bucket_name,
    #                prefix=pipeline_yaml_path,
    #                dl_dir="./")

    # index_file_name = index_faiss_file_path.split('/')[-1]
    # os.chdir(artifact_path)

    # new_document_store = FAISSDocumentStore.load(index_file_name)
    # reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

    # retriever = EmbeddingRetriever(
    #     document_store=new_document_store,
    #     embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    #     model_format="sentence_transformers",
    #     use_gpu=False,
    # )

    # extractive_pipe = ExtractiveQAPipeline(reader, retriever)

    # #### TESTING THE TRAINING COMPONENT ###
    # extractive_pred = extractive_pipe.run(
    #     query="who is the father of Athena?", params={"Retriever": {"top_k": 3}, "Reader": {"top_k": 3}}
    #     )

    # for answer in extractive_pred['answers']:
    #     if answer.score > 0.1:
    #         print('answer:::', answer.answer)
    #         print('context:::', answer.context)
    #         print('confidence:::', answer.score)
    #         print('\n')
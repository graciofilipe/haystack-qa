from fastapi import FastAPI
from pydantic import BaseModel
import os
import subprocess
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import FARMReader, EmbeddingRetriever
from haystack.pipelines import ExtractiveQAPipeline
from datetime import datetime

from google.cloud import storage


currentDateAndTime = datetime.now()

print("runniung the mains that were created at", currentDateAndTime)

ARTIFACT_PATH = "myth/model/" 
index_faiss_file_path = "my_faiss_index.faiss"
index_json_file_path = "my_faiss_index.json"
document_db_path = "faiss_document_store.db"
pipeline_yaml_path = "pipe.yaml"
bucket_name="filipegracio-haystack"


def download_files(bucket_name="filipegracio-haystack",
                prefix=None,
                dl_dir="./"):
    
    from google.cloud import storage
    from pathlib import Path
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        file_split = blob.name.split("/")
        directory = "/".join(file_split[0:-1])
        Path(directory).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(dl_dir + blob.name)


download_files(bucket_name=bucket_name,
            prefix=ARTIFACT_PATH + index_faiss_file_path,
            dl_dir="./")

download_files(bucket_name=bucket_name,
            prefix=ARTIFACT_PATH + index_json_file_path,
            dl_dir="./")

download_files(bucket_name=bucket_name,
            prefix= ARTIFACT_PATH + document_db_path,
            dl_dir="./")

download_files(bucket_name=bucket_name,
            prefix=ARTIFACT_PATH + pipeline_yaml_path,
            dl_dir="./")

index_file_name = index_faiss_file_path.split('/')[-1]
os.chdir(ARTIFACT_PATH)

new_document_store = FAISSDocumentStore.load(index_file_name)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

retriever = EmbeddingRetriever(
document_store=new_document_store,
embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
model_format="sentence_transformers",
use_gpu=False,
)

extractive_pipe = ExtractiveQAPipeline(reader, retriever)

##
class QuestionQuery(BaseModel):
    query: str
    top_k: int

##

app = FastAPI()

@app.get("/hello")
async def return_hello():
    return "Hellow \n"

@app.post("/question/")
async def question(question: QuestionQuery):

    extractive_pred = extractive_pipe.run(
            query=question.query,
            params={"Retriever": {"top_k": question.top_k}, "Reader": {"top_k": question.top_k}}
        )

    dict_of_responses = {}
    counter = 0
    for answer in extractive_pred['answers']:
        if answer.score > 0.7:
                dict_of_responses[str(counter)] = answer.answer
                counter += 1

    return dict_of_responses

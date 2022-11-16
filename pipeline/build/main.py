from fastapi import FastAPI
from pydantic import BaseModel
import os
import subprocess
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import FARMReader, EmbeddingRetriever
from haystack.pipelines import ExtractiveQAPipeline
from datetime import datetime

from google.cloud import storage


# we'll need this utility to download the model files
def download_files(bucket_name=None,
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




# At deploy time, we will pass an env-var to the Cloud Run environment. That's the bucket path to the model files
# we need those to be loaded by the app 
ARTIFACT_PATH = os.getenv('artifact_path')
bucket_name=os.getenv('bucket_name')


# these are the standard file names for the model to be loaded. 
index_faiss_file_path = "my_faiss_index.faiss"
index_json_file_path = "my_faiss_index.json"
document_db_path = "faiss_document_store.db"


## this downloads the model artifacts to the app run environment
download_files(bucket_name=bucket_name,
            prefix=ARTIFACT_PATH + index_faiss_file_path,
            dl_dir="./")

download_files(bucket_name=bucket_name,
            prefix=ARTIFACT_PATH + index_json_file_path,
            dl_dir="./")

download_files(bucket_name=bucket_name,
            prefix= ARTIFACT_PATH + document_db_path,
            dl_dir="./")

index_file_name = index_faiss_file_path.split('/')[-1]
os.chdir(ARTIFACT_PATH)

# now we grab the model that we just downloaded
new_document_store = FAISSDocumentStore.load(index_file_name)

# the reader needs to be defined to read the documents (no GPU at serving time)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

# the retriever is based on the document store that we just downloaded
retriever = EmbeddingRetriever(
document_store=new_document_store,
embedding_model="sentence-transformers/multi-qa-mpnet-base-cos-v1",
model_format="sentence_transformers",
use_gpu=False,
)

# the pipeline of the reader plus retriever is what will provide the answer
extractive_pipe = ExtractiveQAPipeline(reader, retriever)

## FAST API SPECIFIC CODE ##

# A class that will carry the payload to the POST section of the app
class QuestionQuery(BaseModel):
    query: str
    cut_off: float

##

app = FastAPI()

# a hello world utility
@app.get("/hello")
async def return_hello():
    return "Hello \n"

# the section that takes in the payload with the question and returns the answers
@app.post("/question/")
async def question(payload: QuestionQuery):

    # this global var defined above comes from the Haystack model we just loaded 
    extractive_pred = extractive_pipe.run(
            query=payload.query,
            params={"Retriever": {"top_k":10}, "Reader": {"top_k": 5}} # recommended params 
        )

    # capture some of the main components of the answer and send as a response to the request
    dict_of_responses = {}
    counter = 0
    for answer in extractive_pred['answers']:
        if answer.score > payload.cut_off:
                  dict_of_responses["answer " + str(counter)] = {
                      "score": answer.score,
                      "answer": answer.answer, 
                      "doc": answer.meta["name"],
                      "context": answer.context}
        counter += 1

    return dict_of_responses

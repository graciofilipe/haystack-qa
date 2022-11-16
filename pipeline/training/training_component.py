from kfp.v2.dsl import component
from typing import NamedTuple
import os

project_name = os.getenv("PROJECT_NAME")

@component(
    base_image="europe-docker.pkg.dev/"+project_name+"/haystack-docker-repo/haystack-training:latest",
)
def training_comp(
    project_name: str=None,
    bucket_name: str = None,
    data_path: str = None,
    artifact_path: str = None
    ) -> NamedTuple(
    "Outputs",
    [
        ("index_faiss_file_path", str),  
        ("index_json_file_path", str),  
        ("document_db_path", str),
        ("artifact_path", str),
        ("training_complete", str)
    ],
):
    '''
    bucket_name: where all the data and artifacts are or will be stored
    data_path: where in the bucket is the data to be trained on (txt files)
    artifact_path: where to write the model files after training
    '''

    from google.cloud import storage

    from pathlib import Path
    import logging

    from haystack.utils import convert_files_to_docs, launch_es
    from haystack.document_stores import FAISSDocumentStore
    from haystack.nodes import PreProcessor, EmbeddingRetriever

    ## AUX FUNCTIONS ##
    def download_files(bucket_name,
                       prefix,
                       dl_dir):

        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            file_split = blob.name.split("/")
            directory = "/".join(file_split[0:-1])
            Path(directory).mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(blob.name)
    

    def upload_blob(bucket_name, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print(
            f"File {source_file_name} uploaded to {destination_blob_name}."
        )


    ## MAIN WORK ##
    # needs to bring the data for model training
    download_files(bucket_name=bucket_name, prefix=data_path, dl_dir=data_path)

    # a search search server
    launch_es()

    # imports the local files to memory
    all_docs = convert_files_to_docs(dir_path=data_path)

    # process the documents into small docs that are easier to search
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=100,
        split_respect_sentence_boundary=True,
    )
    docs = preprocessor.process(all_docs)

    # writing the documents int he document store
    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat",  similarity="cosine")
    document_store.write_documents(docs)

    # what will do the embedding
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-cos-v1",
        model_format="sentence_transformers",
    )
    
    document_store.update_embeddings(retriever=retriever)

    # Next we need to save the result of this work
    # this section saves the model locally and then uploads everything to "artifact_path" in "bucket_name"
    document_store.save("my_faiss_index.faiss")

    upload_blob(bucket_name=bucket_name, 
                source_file_name="my_faiss_index.faiss", 
                destination_blob_name=artifact_path + "my_faiss_index.faiss")

    upload_blob(bucket_name=bucket_name, 
                source_file_name="my_faiss_index.json", 
                destination_blob_name=artifact_path + "my_faiss_index.json")

    upload_blob(bucket_name=bucket_name, 
                source_file_name="faiss_document_store.db", 
                destination_blob_name=artifact_path + "faiss_document_store.db")


    # outputs that get passed over to the next component
    return(artifact_path + "my_faiss_index.faiss",
    artifact_path + "my_faiss_index.json", 
    artifact_path + "faiss_document_store.db",
    artifact_path,
    "true")
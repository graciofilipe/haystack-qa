from kfp.v2.dsl import component
from typing import NamedTuple



@component(
    base_image="europe-docker.pkg.dev/filipegracio-ai-learning/haystack-docker-repo/haystack-training:tag1",
)
def training_comp(
    bucket_name: str ="filipegracio-haystack",
    data_path: str ="myth/data",
    artifact_path: str = "myth/model/"
    ) -> NamedTuple(
    "Outputs",
    [
        ("index_faiss_file_path", str),  
        ("index_json_file_path", str),  
        ("document_db_path", str),
        ("pipeline_yaml_path", str)
    ],
):

    from google.cloud import storage
    from pathlib import Path
    import logging


    from haystack.utils import convert_files_to_docs, launch_es
    from haystack.document_stores import FAISSDocumentStore
    from haystack.nodes import PreProcessor, EmbeddingRetriever, FARMReader
    from haystack.pipelines import ExtractiveQAPipeline

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
    
    from google.cloud import storage


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




    ## MAIN WORK ##
    download_files(bucket_name=bucket_name, prefix=data_path, dl_dir=data_path)

    launch_es()

    all_docs = convert_files_to_docs(dir_path=data_path)

    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=100,
        split_respect_sentence_boundary=True,
    )
    docs = preprocessor.process(all_docs)

    print(f"n_files_input: {len(all_docs)}\nn_docs_output: {len(docs)}")

    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat",  similarity="cosine")
    document_store.write_documents(docs)


    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        model_format="sentence_transformers",
    )

    document_store.update_embeddings(retriever=retriever)

    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    extractive_pipe = ExtractiveQAPipeline(reader, retriever)

    extractive_pipe.save_to_yaml(path='pipe.yaml')
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

    upload_blob(bucket_name=bucket_name, 
                source_file_name="pipe.yaml", 
                destination_blob_name=artifact_path + "pipe.yaml")


    #### TESTING THE TRAINING COMPONENT ###
    # extractive_pred = extractive_pipe.run(
    #     query="who is the father of Athena?", params={"Retriever": {"top_k": 3}, "Reader": {"top_k": 3}}
    #     )

    # for answer in extractive_pred['answers']:
    #     if answer.score > 0.1:
    #         print('answer:::', answer.answer)
    #         print('context:::', answer.context)
    #         print('confidence:::', answer.score)
    #         print('\n')

    return(artifact_path + "my_faiss_index.faiss",
    artifact_path + "my_faiss_index.json", 
    artifact_path + "faiss_document_store.db",
    artifact_path + "pipe.yaml")
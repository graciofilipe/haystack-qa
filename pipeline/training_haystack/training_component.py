from kfp.v2.dsl import component



@component(
    base_image="europe-docker.pkg.dev/filipegracio-ai-learning/haystack-docker-repo/haystack-training:tag1",
)
def training_comp(
    bucket_name: str ="filipegracio-haystack",
    data_path: str ="data/myth/"
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


    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        model_format="sentence_transformers",
    )

    info.logging('update embedings')
    document_store.update_embeddings(retriever=retriever)
    info.logging('updateDDD embedings')


    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)


    extractive_pipe = ExtractiveQAPipeline(reader, retriever)

    extractive_pipe.save_to_yaml(path='pipe.yaml')
    document_store.save("my_faiss_index.faiss")
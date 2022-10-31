import google.cloud.aiplatform as aip
from google_cloud_pipeline_components.experimental.custom_job import utils
from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component
from google.cloud import storage
from python_utils import download_files


@component(
    base_image="europe-docker.pkg.dev/filipegracio-ai-learning/haystack-docker-repo/haystack-training:tag1",
)
def training(
    bucket_name: str ="filipegracio-haystack",
    data_path: str ="data/myth/"
    ):
    from haystack.utils import convert_files_to_docs, launch_es
    from haystack.document_stores import FAISSDocumentStore
    from haystack.nodes import PreProcessor, EmbeddingRetriever, FARMReader
    from haystack.pipelines import ExtractiveQAPipeline

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


    document_store.update_embeddings(retriever=retriever)

    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)


    extractive_pipe = ExtractiveQAPipeline(reader, retriever)

    extractive_pipe.save_to_yaml(path='pipe.yaml')
    document_store.save("my_faiss_index.faiss")
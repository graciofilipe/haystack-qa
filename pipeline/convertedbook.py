from haystack.utils import launch_es

launch_es()

from haystack.utils import convert_files_to_docs
all_docs = convert_files_to_docs(dir_path="data/myth/")


from haystack.nodes import PreProcessor

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

from haystack.document_stores import FAISSDocumentStore
document_store = FAISSDocumentStore(faiss_index_factory_str="Flat",  similarity="cosine")


document_store.write_documents(docs)


from haystack.nodes import EmbeddingRetriever

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_format="sentence_transformers",
)


document_store.update_embeddings(retriever=retriever)


from haystack.nodes import FARMReader


# Load a  local model or any of the QA models on
# Hugging Face's model hub (https://huggingface.co/models)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

from haystack.pipelines import ExtractiveQAPipeline
extractive_pipe = ExtractiveQAPipeline(reader, retriever)

extractive_pipe.save_to_yaml(path='pipe.yaml')
document_store.save("my_faiss_index.faiss")


from haystack.utils import print_answers

extractive_pred = extractive_pipe.run(
    query="who is the father of Athena?", params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 5}}
)
for answer in extractive_pred['answers']:
    if answer.score > 0.1:
        print('answer:::', answer.answer)
        print('context:::', answer.context)
        print('confidence:::', answer.score)
        print('\n')
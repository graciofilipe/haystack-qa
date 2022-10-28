from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import FARMReader, EmbeddingRetriever
from haystack.pipelines import ExtractiveQAPipeline
import argparse


new_document_store = FAISSDocumentStore.load("my_faiss_index.faiss")
# assert new_document_store.faiss_index_factory_str == "Flat"

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

retriever = EmbeddingRetriever(
    document_store=new_document_store,
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_format="sentence_transformers",
    use_gpu=False,
)


extractive_pipe = ExtractiveQAPipeline(reader, retriever)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="the string you use here")
    args = parser.parse_args()

    extractive_pred = extractive_pipe.run(
        query=args.question,
        params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 5}}
    )
    for answer in extractive_pred['answers']:
        if answer.score > 0.5:
            print('answer:::', answer.answer)
            print('context:::', answer.context)
            print('confidence:::', answer.score)
            print('\n')
            
from fastapi import FastAPI
from pydantic import BaseModel
import os
import subprocess
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

##
class QuestionQuery(BaseModel):
    query: str
    top_k: int

##

app = FastAPI()

@app.get("/hello")
async def return_hello():
    return "Hellow"

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

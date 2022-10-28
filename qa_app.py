import os
import subprocess
from flask import Flask, request, make_response
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

print("Started...")
app = Flask(__name__)

@app.route('/')
def func1():
    print("Running")
    return "Hello World"

@app.route('/exec', methods=['POST'])
def route_exec():
    command = request.data.decode('utf-8')
    try:
        extractive_pred = extractive_pipe.run(
            query="who is the father of Athena?",
            params={"Retriever": {"top_k": 1}, "Reader": {"top_k": 1}}
        )
        
        # completedProcess = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10, universal_newlines=True)
        for answer in extractive_pred['answers']:
             if answer.score > 0.90:
                 response = make_response(answer.answer, 200)

        return response

    except subprocess.TimeoutExpired:
        response = make_response("Timedout", 400)
        response.mimetype = "text/plain"
        return response
    return "/exec"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get('PORT', 8080)))
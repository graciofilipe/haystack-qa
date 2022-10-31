# syntax=docker/dockerfile:1
FROM python:3.10
ENV PYTHONUNBUFFERED True

RUN pip install --upgrade pip
RUN pip install git+https://github.com/deepset-ai/haystack.git#egg=farm-haystack[colab,ocr]
RUN pip install farm-haystack[faiss]
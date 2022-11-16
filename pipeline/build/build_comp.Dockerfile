# syntax=docker/dockerfile:1
FROM python:3.10
ENV PYTHONUNBUFFERED True

RUN pip install google-cloud-storage
RUN pip install gcloud
RUN pip install google-api
RUN pip install google-api-core
RUN pip install google-cloud-build
      
# syntax=docker/dockerfile:1
FROM europe-docker.pkg.dev/filipegracio-ai-learning/haystack-docker-repo/haystack-deploy:tag1
ENV PYTHONUNBUFFERED True

RUN pip install fastapi
RUN pip install pydantic
RUN pip install gcloud
COPY . .

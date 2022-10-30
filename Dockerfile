# syntax=docker/dockerfile:1
FROM python:3.10
ENV PYTHONUNBUFFERED True



COPY . .
RUN sh install.sh


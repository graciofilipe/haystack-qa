# syntax=docker/dockerfile:1
FROM python:3.10
COPY . .
RUN sh install.sh
#RUN python "who is the father of Athena?"
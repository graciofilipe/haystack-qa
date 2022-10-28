# syntax=docker/dockerfile:1
FROM python:3.10
COPY . .
RUN sh install.sh

# Install production dependencies.
RUN pip install Flask gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
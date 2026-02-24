# Custom Airflow image with pipeline dependencies (datasets, TFDV, etc.)
# Build with: docker build --platform linux/amd64 -t live-speech-airflow .
# (TFDV only has x86_64 wheels; amd64 runs via Rosetta on Apple Silicon)
FROM apache/airflow:2.10.4-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

USER airflow
WORKDIR /opt/airflow

COPY requirements-docker.txt /opt/airflow/requirements-docker.txt
# TFDV first (pins pyarrow<11); then pipeline deps (datasets etc. may upgrade pyarrow)
RUN pip install --no-cache-dir tensorflow-data-validation \
    && pip install --no-cache-dir -r /opt/airflow/requirements-docker.txt

FROM --platform=linux/amd64 apache/airflow:2.9.3-python3.9

USER root
RUN apt-get update && apt-get install -y build-essential python3-dev && apt-get clean

USER airflow
RUN pip install --no-cache-dir \
    tensorflow-data-validation==1.15.0 \
    pandas==1.5.3 \
    pyarrow==10.0.1 \
    "protobuf>=3.20,<5.0" \
    tensorflow==2.15.1 \
    dill==0.3.8

RUN pip install --no-cache-dir \
    datasets==2.19.0 \
    langdetect==1.0.9 \
    tqdm==4.66.0 \
    google-cloud-storage==2.16.0 \
    ftfy==6.2.0 \
    sacrebleu==2.4.0 \
    "dvc[gs]==3.50.0"
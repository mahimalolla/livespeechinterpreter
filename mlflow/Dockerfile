FROM python:3.10-slim

RUN pip install mlflow==2.12.1 google-cloud-storage boto3 --no-cache-dir

ENV MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=gs://livespeechinterpreter-training/mlflow-artifacts

EXPOSE 8080

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "gs://livespeechinterpreter-training/mlflow-artifacts", \
     "--serve-artifacts"]
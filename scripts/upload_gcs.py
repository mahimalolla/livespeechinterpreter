import os
import json
from google.cloud import storage

GCS_BUCKET = "livespeechinterpreter"
GCS_PREFIX = "datasets/v2_approved/"
KEY_PATH = "/opt/airflow/gcp-key.json"

def upload_to_gcs():
    print(f"Connecting to GCS bucket: {GCS_BUCKET}...")
    client = storage.Client.from_service_account_json(KEY_PATH)
    bucket = client.bucket(GCS_BUCKET)

    files_to_upload = [
        "/opt/airflow/data/processed/train.jsonl",
        "/opt/airflow/data/processed/val.jsonl",
        "/opt/airflow/data/processed/test.jsonl",
        "/opt/airflow/reports/schema.json",
        "/opt/airflow/reports/tfdv_stats_summary.json",
        "/opt/airflow/reports/anomalies.json",
        "/opt/airflow/reports/dataset_approval.json",
    ]

    # Add slices
    slices_dir = "/opt/airflow/data/processed/slices"
    for f in os.listdir(slices_dir):
        if f.endswith(".jsonl"):
            files_to_upload.append(os.path.join(slices_dir, f))

    for local_path in files_to_upload:
        if not os.path.exists(local_path):
            print(f"WARNING: {local_path} not found, skipping...")
            continue
        blob_name = GCS_PREFIX + os.path.basename(local_path)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        size_mb = os.path.getsize(local_path) / 1024 / 1024
        print(f"Uploaded {os.path.basename(local_path)} ({size_mb:.1f}MB) → gs://{GCS_BUCKET}/{blob_name}")

    print(f"\nAll files uploaded to gs://{GCS_BUCKET}/{GCS_PREFIX}")
    print("Node 9 complete.")
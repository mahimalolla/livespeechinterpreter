import json
import datetime
import os

MIN_TRAIN = 10000

def approve_dataset():
    print("Counting records in splits...")

    train_count = 0
    with open("/opt/airflow/data/processed/train.jsonl") as f:
        for line in f:
            if line.strip():
                train_count += 1

    val_count = 0
    with open("/opt/airflow/data/processed/val.jsonl") as f:
        for line in f:
            if line.strip():
                val_count += 1

    test_count = 0
    with open("/opt/airflow/data/processed/test.jsonl") as f:
        for line in f:
            if line.strip():
                test_count += 1

    if train_count < MIN_TRAIN:
        raise ValueError(f"Train set too small: {train_count} < {MIN_TRAIN}")

    approval = {
        "approved": True,
        "timestamp": str(datetime.datetime.utcnow()),
        "train_records": train_count,
        "val_records": val_count,
        "test_records": test_count,
        "version": "v2_approved",
    }

    json.dump(approval, open("/opt/airflow/reports/dataset_approval.json", "w"), indent=2)
    print(f"Dataset APPROVED:")
    print(json.dumps(approval, indent=2))
    print("Node 8 complete.")

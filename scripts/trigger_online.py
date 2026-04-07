import json

def trigger_online_pipeline():
    approval = json.load(open("/opt/airflow/reports/dataset_approval.json"))

    trigger_payload = {
        "gcs_train": "gs://translation-training-data/datasets/v2_approved/train.json",
        "gcs_val": "gs://translation-training-data/datasets/v2_approved/val.json",
        "gcs_schema": "gs://translation-training-data/datasets/v2_approved/schema.json",
        "dataset_version": approval["version"],
        "train_records": approval["train_records"],
    }

    json.dump(trigger_payload, open("/opt/airflow/reports/trigger_payload.json", "w"), indent=2)
    print("Trigger payload ready for Phase 2:")
    print(json.dumps(trigger_payload, indent=2))
    print("Node 10 complete. Offline pipeline finished.")
# DAG

This directory contains the Apache Airflow DAG for the offline data pipeline.

Prepares and validates bilingual (EN↔ES) training data before uploading to Google Cloud Storage for model fine-tuning.

---

## Airflow Pipeline

```
            ┌──────────────────────────────┐
            │  node0_dvc_pull_or_download  │
            └──────────────┬───────────────┘
                           │
                 branch_after_node0
                /                 \
        [cache hit]           [no cache]
               │                   │
               │        ┌──────────▼──────────────┐
               │        │ node1_download_datasets │
               │        └──────────┬──────────────┘
               │                   │
        ┌──────▼───────────────────▼──────┐
        │        node2_preprocess         │
        └──────────────────┬──────────────┘
                           │
        ┌──────────────────▼──────────────┐
        │    node3_train_val_test_split   │
        └──────────────────┬──────────────┘
                           │
        ┌──────────────────▼──────────────┐
        │       node4_ge_statistics       │
        └──────────────────┬──────────────┘
                           │
        ┌──────────────────▼──────────────┐
        │      node5_schema_inference     │
        └──────────────────┬──────────────┘
                           │
        ┌──────────────────▼──────────────┐
        │     node6_anomaly_detection     │
        └──────────────────┬──────────────┘
                           │
        ┌──────────────────▼──────────────┐
        │      node6b_bias_detection      │
        └──────────────────┬──────────────┘
                           │
        ┌──────────────────▼──────────────┐
        │      node7_dataset_slicing      │
        └──────────────────┬──────────────┘
                           │
        ┌──────────────────▼──────────────┐
        │      node8_dataset_approval     │
        └──────────────────┬──────────────┘
                           │
        ┌──────────────────▼──────────────┐
        │       node9_upload_to_gcs       │
        └──────────────────┬──────────────┘
                           │
        ┌──────────────────▼──────────────┐
        │         node9b_dvc_push         │
        └──────────────────┬──────────────┘
                           │
        ┌──────────────────▼──────────────┐
        │   node10_trigger_online_pipeline│
        └─────────────────────────────────┘
```
---


## Node Reference

| Task ID | Script | Description |
|---|---|---|
| `node0_dvc_pull_or_download` | `dvc_pull.py` | Attempts DVC cache restore; returns `True` if raw data is already present |
| `branch_after_node0` | *(inline)* | Skips download if DVC cache hit, otherwise runs node1 |
| `node1_download_datasets` | `download_datasets.py` | Downloads OPUS-100 + EuroParl EN↔ES parallel corpora |
| `node2_preprocess` | `preprocess.py` | Cleans text and formats records into instruction-tuning format |
| `node3_train_val_test_split` | `split_dataset.py` | Splits data 80 / 10 / 10 (train / val / test) |
| `node4_ge_statistics` | `ge_stats.py` | Generates Great Expectations statistical profile |
| `node5_schema_inference` | `schema_inference.py` | Infers TFDV schema from training split |
| `node6_anomaly_detection` | `anomaly_detection.py` | Flags out-of-distribution samples against the inferred schema |
| `node6b_bias_detection` | `bias_detection.py` | Slice-level bias analysis (medical/legal, short/long, EN→ES/ES→EN) |
| `node7_dataset_slicing` | `dataset_slicing.py` | Creates named slices: `medical`, `legal`, `en_to_es`, `es_to_en`, `short`, `long` |
| `node8_dataset_approval` | `dataset_approval.py` | Manual approval gate - pipeline halts until data is approved |
| `node9_upload_to_gcs` | `upload_gcs.py` | Uploads approved data to gcs |
| `node9b_dvc_push` | `dvc_push.py` | Pushes dataset version to DVC remote (GCS) |
| `node10_trigger_online_pipeline` | `trigger_online.py` | Triggers online training pipeline |

---

## Branching Logic

`node0` checks whether raw data already exists (via DVC cache restore).

- **Cache hit** (`True`) → skips `node1_download_datasets`, jumps straight to `node2_preprocess`
- **No cache** (`False`) → runs `node1_download_datasets` then `node2_preprocess`

`node2_preprocess` uses `trigger_rule="none_failed_min_one_success"` so it runs correctly from either branch.

## Running Locally

```bash
# Start Airflow via Docker Compose (from project root)
docker compose up -d --no-build

# Airflow UI
open http://localhost:8080   # admin / admin

# Trigger the DAG manually from the UI or CLI
airflow dags trigger offline_translation_pipeline

# Tear down
docker compose down
```

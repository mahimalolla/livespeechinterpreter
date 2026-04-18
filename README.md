# VISTA Live Speech Interpreter - End-to-End MLOps Translation System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![GCP](https://img.shields.io/badge/GCP-Cloud-orange)

---

## Overview

**VISTA Live Speech Interpreter** is a real-time **speech-to-speech translation system** (English ↔ Spanish) specialized for **medical and legal domains** built with a production-grade MLOps pipeline from raw data ingestion through fine-tuning, deployment, and live inference.

It enables seamless bilingual communication in **medical and legal domains** using:

- Microphone → Google STT
- Fine-tuned LLM (Gemma 3 4B with QLoRA)
- Google TTS → Speaker

**Live API:** `https://translation-api-1050963407386.us-central1.run.app`

**MLflow Tracking:** `https://mlflow-server-1050963407386.us-central1.run.app`

---

## How It Works

```
Microphone
    └── Google Speech-to-Text (STT)
            └── Fine-tuned Gemma 3 4B (QLoRA) — FastAPI on Cloud Run
                    └── Google Text-to-Speech (TTS)
                            └── Speaker
```

The system supports two translation directions (`en_to_es`, `es_to_en`) and two domains (`medical`, `legal`), with domain-aware output at inference time.

---

## System Architecture

### Data Pipeline (Apache Airflow + Docker)

The `offline_translation_pipeline` Airflow DAG processes and validates training data before uploading to Google Cloud Storage. Runs as a directed graph with a DVC-aware branch:

```
DVC pull/download
    └── [if DVC cache hit] → skip download
    └── [if no cache]      → download datasets (OPUS-100, EuroParl)
         ↓
    Preprocess → Train/Val/Test split (80/10/10)
         ↓
    GE statistics → Schema inference → Anomaly detection → Bias detection
         ↓
    Dataset slicing → Dataset approval gate
         ↓
    Upload to GCS → DVC push → Trigger online pipeline
```

Scripts in `scripts/`:

| Script | Purpose |
|---|---|
| `download_datasets.py` | Pull OPUS-100 + EuroParl EN↔ES pairs |
| `preprocess.py` | Clean and format into instruction tuning records |
| `split_dataset.py` | 80/10/10 train/val/test split |
| `ge_stats.py` | Great Expectations statistical profiling |
| `schema_inference.py` | TFDV schema generation |
| `anomaly_detection.py` | Flag out-of-distribution samples |
| `bias_detection.py` | Slice-level bias check |
| `dataset_slicing.py` | Create slices: medical, legal, en_to_es, es_to_en, short_sents, long_sents |
| `dataset_approval.py` | Approval gate before upload |
| `upload_gcs.py` | Push approved data to `gs://livespeechinterpreter/datasets/v2_approved` |
| `sensitivity_analysis.py` | Post-training data + hyperparameter sensitivity report |
| `evaluate_model.py` | BLEU + slice-level bias evaluation |
| `register_existing_model_mlflow.py` | Register a GCS adapter in the MLflow Model Registry |
| `vertex_training_mlflow.py` | Register model that ran on Vertex AI into MLflow |

---

### Model Training (Vertex AI)

Training runs on Vertex AI using `training/train.py`. Fine-tunes `google/gemma-3-4b-it` with QLoRA on a T4 GPU. The resulting adapter (~131 MB) is uploaded to GCS on completion and the run is logged to MLflow.


| Setting | Value |
|---|---|
| Base model | `google/gemma-3-4b-it` |
| Fine-tuning | QLoRA (4-bit NF4 quantization) |
| LoRA rank | r=16, alpha=32, dropout=0.05 |
| Learning rate | 5e-5 (critical - 2e-4 causes loss explosion) |
| Attention | `eager` (required - `sdpa` causes gradient instability on Gemma3) |
| Batch size | 4, gradient accumulation 4 (effective batch 16) |
| Sequence length | 128 (256 OOMs on T4) |
| Packing | enabled (2× throughput) |
| Max training samples | 50,000 train / 5,000 val |
| Current adapter | `gs://livespeechinterpreter-training/models/gemma3-4b-translation-v17` |
| Adapter size | ~131 MB |

**Model selection thresholds:**
- BLEU ≥ 25
- Bias deviation < 30%

---

### Inference API (FastAPI + Cloud Run)

`inference/main.py` — deployed to Cloud Run at startup, loads the LoRA adapter from GCS asynchronously. Every request is logged to BigQuery.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Root / status |
| `GET` | `/health` | Returns `model_loaded` and `model_ready` flags |
| `POST` | `/translate` | Domain-aware EN↔ES translation |

**Request schema:**
```json
{
  "text": "The patient requires immediate surgery.",
  "direction": "en_to_es",
  "domain": "medical"
}
```

**BigQuery logging:** Every inference call is logged to BigQuery (`mlops-489703.translation_monitoring.inference_logs`) with: timestamp, input/output text, direction, domain, latency_ms, input/output word counts.

---

### CI/CD Pipeline (Cloud Build)

`cloudbuild.yaml` — triggered on every GitHub push. Steps are change-aware (only rebuild images if relevant directories changed):

1. **lint-scripts** — pyflakes on `scripts/` and `dags/`
2. **run-tests** — pytest on `tests/` (if present)
3. **build-training-image** — builds only if `training/` changed → `gemma-trainer:$SHA`
4. **push-training-image** — pushes to Artifact Registry
5. **build-inference-image** — builds only if `inference/` changed → `inference-api:$SHA`
6. **push-inference-image** — pushes to Artifact Registry
7. **evaluate-model** — only if `training/` or `scripts/` changed; runs BLEU + bias eval against v17 adapter; logs to MLflow
8. **deploy-cloud-run** — only if `inference/` changed; deploys to Cloud Run (16 GiB, 4 CPU, min 1 instance)
9. **log-build** — records build metadata to BigQuery (`translation_monitoring.cicd_builds`)
10. **notify-email** — sends Gmail alert via Secret Manager app password

---

### 5. Frontend (React + TypeScript)

Located in `front_end/`. Built with Vite, TailwindCSS, shadcn/ui, and React Query. Containerized with Docker + nginx for deployment.

---

## Experiment Tracking (MLflow)

All training runs, evaluations, and analyses land in the `gemma3-translation` experiment.

| Run type | Script | Run name pattern |
|---|---|---|
| Live training | `training/train.py` | `gemma3-4b-qlora-YYYYMMDD_HHmmSS` |
| Vertex AI model training | `scripts/vertex_training_mlflow.py` | `vertex-train-backfill-YYYYMMDD_HHmmSS` |
| Sensitivity analysis | `scripts/sensitivity_analysis.py` | `sensitivity-analysis` |
| Model evaluation | `scripts/evaluate_model.py` | logged per CI run |

---

## Sensitivity Analysis

`scripts/sensitivity_analysis.py` reads the approved data slices from GCS and measures:

- **Length sensitivity** — output/input word ratio across short (≤10), medium (10-25), long (>25 word) inputs
- **Domain sensitivity** — vocabulary diversity (type-token ratio) in medical vs. legal outputs
- **Direction sensitivity** — average output length for EN→ES vs. ES→EN
- **Hyperparameter sensitivity** — documented findings from v1-v17 experiments (lr, LoRA rank, attention impl, packing, seq length)

Results are logged to MLflow and saved to `reports/sensitivity_report.json`.

**Key findings from training history:**
- Learning rate is HIGH sensitivity — 4× reduction (2e-4 → 5e-5) fixed training stability
- Attention implementation is HIGH sensitivity — `eager` required for Gemma3
- Sequence length is HIGH sensitivity — directly constrains VRAM
- LoRA rank is MEDIUM sensitivity — r=16 gives good quality/cost balance
- Packing is LOW sensitivity — safe to enable for 2× throughput

---

## Bias Detection & Mitigation

**Slices evaluated:** medical vs. legal, short vs. long sentences, EN→ES vs. ES→EN.

**Findings:**
- Legal domain underrepresented in the training corpus
- Sentence length imbalance (skew toward shorter sentences)

**Mitigations applied:**
- Oversampling of legal domain data
- Length-stratified sampling
- Sequence length tuning for better long-sentence coverage

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | Apache Airflow (Docker Compose) |
| Data Versioning | DVC + GCS |
| Data validation | Great Expectations + TensorFlow Data Validation (TFDV) |
| Training | Vertex AI (T4 GPU) |
| Model | Gemma 3 4B + QLoRA (PEFT) |
| Inference Framework | Transformers + TRL |
| API | FastAPI + Uvicorn |
| Deployment | Cloud Run (us-central1) |
| CI/CD | Cloud Build |
| Experiment Tracking | MLflow (Cloud Run) |
| Monitoring | BigQuery |
| Container Registry | Artifact Registry |
| Speech | Google Cloud Speech-to-Text + Text-to-Speech + PyAudio |
| Frontend | React 18, TypeScript, Vite, TailwindCSS, shadcn/ui, React Query |

---

## Running the System

### Pipeline DAGs (Airflow)

```bash
source mlops_env/bin/activate
docker compose up -d --no-build
# Airflow UI: http://localhost:8080
# DAG: offline_translation_pipeline
docker compose down
```

### Live API

```bash
# Health check
curl https://translation-api-1050963407386.us-central1.run.app/health

# Medical EN → ES
curl -X POST https://translation-api-1050963407386.us-central1.run.app/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "The patient requires immediate surgery.", "direction": "en_to_es", "domain": "medical"}'

# Legal ES → EN
curl -X POST https://translation-api-1050963407386.us-central1.run.app/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "El contrato debe ser firmado antes del viernes.", "direction": "es_to_en", "domain": "legal"}'
```

### Vertex Training Run to MLflow

```bash
export MLFLOW_TRACKING_URI=https://mlflow-server-1050963407386.us-central1.run.app
python scripts/vertex_training_mlflow.py \
  --adapter-gcs gs://livespeechinterpreter-training/models/gemma3-4b-translation-v17 \
  --vertex-pipeline-id 4727888025067978752 \
  --vertex-custom-job-id 3534825549954285568 \
  --train-loss-final 6.2549 \
  --train-loss-start 9.35 \
  --train-steps 3588 \
  --train-hours 6.92 \
  --learning-rate 5e-5 \
  --batch-size 4 \
  --epochs 2
```

### Run Sensitivity Analysis

```bash
python scripts/sensitivity_analysis.py
# Logs to MLflow + saves reports/sensitivity_report.json
```

---

## Project Structure

```
.
├── dags/
│   └── offline_pipeline.py       # Airflow DAG (11-node offline pipeline)
├── scripts/
│   ├── download_datasets.py
│   ├── preprocess.py
│   ├── split_dataset.py
│   ├── ge_stats.py
│   ├── schema_inference.py
│   ├── anomaly_detection.py
│   ├── bias_detection.py
│   ├── dataset_slicing.py
│   ├── dataset_approval.py
│   ├── upload_gcs.py
│   ├── sensitivity_analysis.py
│   ├── evaluate_model.py
│   ├── vertex_training_mlflow.py  # Vertex → MLflow backfill
│   ├── register_existing_model_mlflow.py
│   ├── notify_email.py
│   └── tests/
├── training/
│   ├── train.py                   # QLoRA fine-tuning (Vertex AI)
│   ├── Dockerfile
│   └── requirements.txt
├── inference/
│   ├── main.py                    # FastAPI service (Cloud Run)
│   ├── Dockerfile
│   └── requirements.txt
├── data/
│   ├── raw.dvc
│   └── processed.dvc
├── reports/
│   └── sensitivity_report.json
├── dvc.yaml                       # DVC stage definitions
├── cloudbuild.yaml                # CI/CD pipeline
├── docker-compose.yml             # Airflow local setup
├── commands.txt                   # Quick-reference commands
└── requirements.txt
```
---

## Future Enhancements

- Multi-language support (French, Mandarin, etc.)
- Speaker diarization
- Streaming low-latency translation
- Authentication & rate limiting for the API
- Advancement in Frontend UI for better user experience

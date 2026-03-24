#  Live Interpreter --- End-to-End MLOps Translation System

![Python](https://img.shields.io/badge/Python-3.10-blue)\
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)\
![GCP](https://img.shields.io/badge/GCP-Cloud-orange)\
![MLOps](https://img.shields.io/badge/MLOps-End--to--End-purple)\
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

##  Overview

**Live Interpreter** is a real-time **speech-to-speech translation system (English ↔ Spanish)** built with a **production-grade MLOps pipeline**.

It enables seamless bilingual communication in **medical and legal domains** using:

- 🎤 Speech-to-Text (Google STT)\
- 🤖 Fine-tuned LLM (Gemma 3 4B with QLoRA)\
- 🔊 Text-to-Speech (Google TTS)

---

##  End-to-End Pipeline

Mic → Google STT → FastAPI (/translate) → Fine-tuned LLM → Google TTS → Speaker\
---

##  System Architecture

### 🔹 Offline Pipeline (Airflow + Docker)

- Dataset ingestion (OPUS-100, EuroParl)\
- Data preprocessing & instruction formatting\
- Train/Val/Test split (80/10/10)\
- Schema validation (TFDV)\
- Anomaly detection\
- Dataset slicing (domain, direction, sentence length)\
- Bias detection across slices\
- Dataset approval gate\
- Upload to GCS + DVC versioning

---

### 🔹 Model Training (Vertex AI)

- Model: **Gemma 3 4B**\
- Fine-tuning: **QLoRA (4-bit quantization)**\
- Training data: **2.1M EN↔ES sentence pairs**\
- Adapter size: **131MB**

#### Model Selection Criteria\
- BLEU score ≥ 25\
- Bias deviation < 30%

---

### 🔹 Online Inference (FastAPI + Cloud Run)

- `/translate` endpoint (domain-aware translation)\
- Async model loading\
- CPU-based inference (Cloud Run)\
- Logging to BigQuery:\
  - latency\
  - input/output length\
  - domain\
  - translation direction

---

### 🔹 CI/CD Pipeline (Cloud Build)

Triggered automatically on every GitHub push:

- Code linting & tests\
- Model evaluation\
- Bias detection\
- Conditional deployment\
- Logging to BigQuery

---

##  Key Features

-  **End-to-End MLOps Pipeline** (data → training → deployment → monitoring)\
-  **Automated CI/CD with evaluation gates**\
-  **Bias-aware model selection**\
-  **Experiment tracking with MLflow**\
-  **Fully deployed on Google Cloud Platform**\
-  **LLM fine-tuning with QLoRA**\
-  **Real-time inference with FastAPI**\
-  **Containerized architecture (Docker)**\
-  **Dataset validation + anomaly detection**\
-  **BigQuery-based monitoring + logging**

---

##  Model Validation & Evaluation

### Evaluation Metrics\
- BLEU score\
- Latency\
- Output length consistency

### Dataset QA\
- Schema validation\
- Statistical profiling\
- Anomaly detection

### Slice-Based Evaluation\
- Medical vs Legal domain\
- Short vs Long sentences\
- EN → ES vs ES → EN

---

###  Deployment Gates

Model deployment is blocked if:\
- BLEU score < 25\
- Bias deviation > 30%

---

##  Bias Detection & Mitigation

### Slices Evaluated\
- Medical vs Legal\
- Short vs Long sentences\
- Translation direction (EN↔ES)

### Key Findings\
- Legal data underrepresented\
- Sentence length imbalance

### Mitigation Strategies\
- Oversampling legal dataset\
- Length-stratified sampling\
- Increased sequence length

---

##  Experiment Tracking

- Tool: **MLflow (deployed on Cloud Run)**

### Logged Data\
- Hyperparameters\
- Training loss\
- Evaluation metrics\
- Sensitivity analysis

---

##  Sensitivity Analysis

### Performed On\
- Feature impact (length, domain, direction)\
- Hyperparameter tuning effects

### Insights\
- Sequence length significantly impacts translation quality\
- Domain imbalance affects BLEU performance

---

##  Tech Stack

| Layer | Technology |\
|------|------------|\
| Orchestration | Airflow (Docker) |\
| Data Versioning | DVC + GCS |\
| Training | Vertex AI |\
| Model | Gemma 3 4B (QLoRA) |\
| API | FastAPI |\
| Deployment | Cloud Run |\
| CI/CD | Cloud Build |\
| Tracking | MLflow |\
| Monitoring | BigQuery |

---

##  Use Cases

- Healthcare communication\
-  Legal interpretation\
-  Education & public services\
-  Accessibility tools\
-  Multilingual interfaces

---

##  Running the System

### 🔹 Offline Pipeline (Airflow)

```bash\
docker compose up -d

-   Airflow UI: <http://localhost:8080>
-   DAG: `offline_translation_pipeline`
```
* * * * *

### 🔹 API (Hosted)

GET /health\
POST /translate

Example:
```
curl -X POST https://<your-api>/translate \\
-H "Content-Type: application/json" \\
-d '{"text":"Hello","direction":"en_to_es","domain":"medical"}'
```
* * * * *

### 🔹 Live Speech Demo

Mic → STT → API → TTS

* * * * *

 Project Structure
--------------------

.\
├── dags/                  # Airflow DAGs\
├── scripts/               # Data + evaluation scripts\
├── data/                  # Raw + processed datasets\
├── reports/               # Validation + bias reports\
├── inference/             # FastAPI service\
├── training/              # Model training code\
├── docker-compose.yml     # Orchestration setup\
├── requirements.txt\
└── README.md

* * * * *

 Key Highlights
-----------------

-    Production-ready MLOps system
-    Real-time LLM-powered translation
-    Fully automated pipeline
-    Bias-aware model selection
-    Cloud-native deployment
-    End-to-end monitoring

* * * * *

 Future Enhancements
----------------------

-    Multi-language support (French, Mandarin, etc.)
-    Speaker diarization
-    Simultaneous translation (low-latency streaming)
-    Direct speech-to-speech models
-    Expanded domain adaptation
-    Advanced drift detection + auto-retraining
-    Authentication & rate limiting for API
-    Frontend UI for real-time interaction

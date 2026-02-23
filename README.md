Live Speech Interpreter
===========================

**Real-Time Speech-to-Speech Translation System (EN ↔ ES)**\
*MLOps Production Pipeline Project -- Team 5*

* * * * *

Overview
-----------

Live Speech Interpreter is a **production-grade, real-time speech-to-speech translation system** that converts spoken English to spoken Spanish (and vice versa) using a cascaded ML architecture:

ASR (Speech → Text) → NMT (Text → Text) → TTS (Text → Speech)

We selected English–Spanish as the initial language pair because Spanish is one of the most widely spoken languages in the world, with over 500 million speakers globally and significant usage across the United States. In public-sector, healthcare, education, and civic environments, English–Spanish communication gaps are among the most common real-world translation challenges. For the scope of this course project, we intentionally limited our implementation to a single language pair to ensure depth in model optimization, latency engineering, monitoring, and MLOps infrastructure rather than breadth across many languages. In future expansions, the architecture is designed to support additional language pairs (e.g., French, Mandarin) with minimal structural changes, enabling scalable multilingual deployment.

The system is designed with a full **MLOps lifecycle**, including:

-   Data Versioning (DVC)

-   Workflow Orchestration (Airflow)

-   Model Validation (Great Expectations)

-   Bias & Fairness Analysis (Fairlearn)

-   CI/CD Automation

-   Monitoring & Drift Detection

-   Cloud Deployment (GCP)

* * * * *

System Architecture
-----------------------

### 🔹 ML Pipeline Components

| Stage | Model | Purpose |
| --- | --- | --- |
| ASR | Conformer / Wav2Vec2 | Speech → English text |
| NMT | T5 / mT5 | English ↔ Spanish translation |
| TTS | Tacotron2 / FastSpeech2 / VITS | Spanish text → Speech |

* * * * *

### 🔹 MLOps Stack

-   **Cloud:** Google Cloud Platform (GKE, Vertex AI, Cloud Storage)

-   **Containerization:** Docker

-   **CI/CD:** GitHub Actions

-   **Orchestration:** Apache Airflow

-   **Data Versioning:** DVC

-   **Model Tracking:** MLflow

-   **Validation:** Great Expectations

-   **Bias Detection:** Fairlearn

-   **Monitoring:** Cloud Monitoring + Logging

* * * * *

Project Structure
--------------------

project/\
├── dags/                     # Airflow DAGs\
├── scripts/\
│   ├── data/                 # Data acquisition & preprocessing\
│   ├── validate/             # Great Expectations validation\
│   ├── bias/                 # Bias analysis (Fairlearn)\
│   ├── models/               # Model training & inference\
│   └── langchain_pipeline.py # LLM-based translation helper\
├── tests/                    # Pytest unit tests\
├── dvc.yaml                  # DVC pipeline stages\
├── docker-compose.yml        # Airflow local deployment\
├── requirements.txt\
└── README.md

* * * * *

Setup Instructions
---------------------

### 1️⃣ Clone Repository

git clone https://github.com/mahimalolla/livespeechinterpreter.git\
cd livespeechinterpreter

* * * * *

### 2️⃣ Create Virtual Environment

python -m venv venv\
source venv/bin/activate  # Mac/Linux\
venv\Scripts\activate     # Windows

* * * * *

### 3️⃣ Install Dependencies

pip install -r requirements.txt

* * * * *

Data Versioning (DVC)
------------------------

### Initialize DVC

dvc init

### Add Remote (Example: GCS)

dvc remote add -d gcsremote gs://your-bucket-name

### Run Full Pipeline

dvc repro

Pipeline stages:

-   `acquire_data`

-   `preprocess_data`

-   `validate_data`

-   `bias_analysis`

* * * * *

Workflow Orchestration (Airflow)
-----------------------------------

### Run Locally with Docker

docker-compose up

Access Airflow UI:

http://localhost:8080

DAG includes:

-   DVC pull

-   Data acquisition

-   Preprocessing

-   Great Expectations validation

-   Bias analysis

-   Slack alert on failure (optional)

* * * * *

Data Validation 
--------------------------------------

We validate:

-   No null translations

-   Source & target length ratio (0.5--2.0)

-   Language consistency

-   No duplicate rows

-   Minimum text length

Run manually:

python scripts/validate/run_ge_validation.py

Validation report is generated in JSON format.

* * * * *

Bias & Fairness Analysis
---------------------------

We perform domain and length-based slicing using **Fairlearn MetricFrame**.

Slicing dimensions:

-   Domain: medical, legal, casual

-   Sentence length buckets: short, medium, long

Metrics:

-   BLEU Score

-   Translation Error Rate

-   Accuracy by slice

Run manually:

python scripts/bias/bias_analysis.py

* * * * *

 Optimization Strategy
------------------------

We optimize for **RECALL**.

### Why?

In legal and medical environments:

-    Missing a statement (False Negative) → Critical harm

-    Extra translation (False Positive) → Minor inconvenience

Tradeoff:\
We accept occasional background noise translations to ensure no important content is missed.

* * * * *

 Key Performance Metrics
--------------------------

### Model Metrics

-   **ASR Word Error Rate (WER):** < 12%

-   **NMT BLEU Score:** > 35

-   **TTS MOS:** > 3.5

### System Metrics

-   **End-to-End Latency:** < 3 seconds

-   **P95 Latency:** < 3s

-   **System Uptime:** 99.9%

-   **Concurrent Sessions:** 100+

* * * * *

 Monitoring & Drift Detection
-------------------------------

We monitor:

-   ASR confidence score trends

-   Translation correction rate

-   Latency breakdown (ASR/NMT/TTS)

-   GPU utilization

-   Model drift via distribution comparison

-   Budget usage alerts

Alerting:

-    P1: Latency > 5s

-    P2: Error rate > 1%

-    Cost > 80% of monthly budget

* * * * *

 Testing
----------

Run tests:

pytest

Includes:

-   Data validation checks

-   Bias slicing tests

-   Pipeline integrity tests

-   Schema validation tests

* * * * *

 Security
-----------

-   TLS 1.3 encryption

-   Encrypted storage (GCP KMS)

-   IAM-based access control

-   No persistent audio storage without user consent

* * * * *

 Project Timeline
--------------------

| Phase | Deliverable |
| --- | --- |
| Phase 1 | Infrastructure + Data Setup |
| Phase 2 | Model Fine-Tuning |
| Phase 3 | Integration + Optimization |
| Phase 4 | Deployment + Monitoring |

* * * * *

 Acceptance Criteria
----------------------

-   Fully automated CI/CD pipeline

-   Infrastructure deployable via Terraform

-   End-to-end system demo under 3 seconds latency

-   Real-time speech translation working in browser

-   Monitoring dashboards active

* * * * *

👥 Team 5
---------

-   Mahima Lolla

-   Suchitra Hole

-   Afrah Fathima Shahabuddin

-   Balaji Sundar Anand Babu

-   Henil Patel

-   Rajiv Praveen

* * * * *

 Future Improvements
----------------------

-   Speaker diarization

-   Direct speech-to-speech model (SeamlessM4T)

-   Voice cloning

-   Multi-language expansion (FR, Mandarin)

-   Simultaneous translation (wait-k policy)

* * * * *

 License
----------

Educational use for MLOps coursework.

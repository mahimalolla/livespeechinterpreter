# Live Speech Interpreter — Data Pipeline
### MLOps Course | Pipeline Orchestration with Apache Airflow
 
---
 
## Table of Contents
 
1. [Project Overview](#1-project-overview)
2. [Environment Setup](#2-environment-setup)
3. [Code Structure](#3-code-structure)
4. [Reproducibility and Data Versioning with DVC](#4-reproducibility-and-data-versioning-with-dvc)
5. [Error Handling and Logging](#5-error-handling-and-logging)
6. [Pipeline DAG Flow](#6-pipeline-dag-flow)
7. [DAG Configuration](#7-dag-configuration)
 
---
 
## 1. Project Overview
 
This repository contains the **data pipeline** for the Live Speech Interpreter project — a real-time English ↔ Spanish speech translation system built on a cascaded ASR → NMT → TTS architecture.
 
The pipeline is fully orchestrated using **Apache Airflow DAGs** and handles the entire workflow automatically:
 
- Fetching versioned data from Google Cloud Storage via DVC
- Downloading raw datasets (LibriSpeech, OPUS EN-ES, domain data)
- Preprocessing audio and text data
- Evaluating data quality
- Saving final outputs and reports
 
**Tech Stack:**
 
| Tool | Purpose |
|------|---------|
| Apache Airflow 2.7.1 | Pipeline orchestration and scheduling |
| DVC | Data versioning and GCP bucket sync |
| HuggingFace Datasets | Downloading LibriSpeech and OPUS |
| Librosa / Soundfile | Audio processing |
| PyTorch / Transformers | Model utilities |
| pytest | Unit testing |
 
---
 
## 2. Environment Setup
 
### Prerequisites
 
Make sure the following are installed on your machine before starting:
 
- [Git](https://git-scm.com/)
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (for DVC remote access)
- [DVC] 
Installation commands :
```bash
conda activate airflow-pipeline
pip install "dvc[gs]"
```

---

### Step 1 — Clone the Repository
 
```bash
git clone https://github.com/mahimalolla/livespeechinterpreter.git
cd livespeechinterpreter
```
 
---
 
### Step 2 — Create the Conda Environment
 
```bash
# Virtual Environment
python -m venv airflow-pipeline
airflow-pipeline\Scripts\activate
```
 
```bash
# Conda
conda create -n airflow-pipeline python=3.10 -y
conda activate airflow-pipeline
```
 
Verify Python version:
 
```bash
python --version
# Expected: Python 3.10.x
```
 
---
 
### Step 3 Setup Apache Airflow
 
Use Airflow to author workflows as directed acyclic graphs (DAGs) of tasks. The Airflow scheduler executes your tasks on an array of workers while following the specified dependencies.
 
References
 
-   Product - https://airflow.apache.org/
-   Documentation - https://airflow.apache.org/docs/
-   Github - https://github.com/apache/airflow
 
#### Installation
 
Prerequisites: You should allocate at least 4GB memory for the Docker Engine (ideally 8GB).
 
Local
 
-   Docker Desktop Running
 
Cloud
 
-   Linux VM
-   SSH Connection
-   Installed Docker Engine - [Install using the convenience script](https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script)
 
---
 
### Step 4 — Install Project Dependencies
 
```bash
pip install -r requirements.txt
```
 
---
 
### Step 5 — Configure DVC Remote (GCP Bucket)
 
```bash
dvc remote add -d gcs_remote gs://your-bucket-name/dvc-cache
dvc remote modify gcs_remote credentialpath /path/to/gcp-credentials.json
```
 
## 3. Code Structure
 
```
livespeechinterpreter/
├── Data-Pipeline/
│   │
│   ├── dags/
│   │   └── live_interpreter_pipeline_dag.py  # main Airflow DAG — defines all tasks
│   │                                          # and their logical connections
│   │
│   ├── scripts/
│   │   ├── paths.py                           # central path config — all data paths
│   │   │                                      # defined in one place, auto-creates folders
│   │   ├── acquire/
│   │   │   ├── acquire_opus.py                # downloads OPUS EN-ES text corpus
│   │   │   ├── acquire_librispeech.py         # downloads LibriSpeech audio dataset
│   │   │   └── acquire_domain_data.py         # pulls domain data from GCP via DVC
│   │   │
│   │   ├── pre_process/
│   │   │   ├── preprocess_audio.py            # cleans audio, normalizes text transcripts
│   │   │   └── preprocess_nmt.py              # cleans EN-ES pairs, filters bad translations
│   │   │
│   │   └── tests/
│   │       ├── test_preprocess_audio.py       # unit tests for audio preprocessing
│   │       └── test_preprocess_nmt.py         # unit tests for NMT preprocessing
│   │
│   ├── data/                                  # all data folders — managed by DVC
│   │   ├── asr/
│   │   │   ├── raw/                           # LibriSpeech raw audio (auto-populated)
│   │   │   └── processed/                     # cleaned audio (auto-populated)
│   │   ├── nmt/
│   │   │   ├── raw/                           # OPUS raw text pairs (auto-populated)
│   │   │   └── processed/                     # cleaned text pairs (auto-populated)
│   │   └── tts/
│   │       ├── raw/
│   │       └── processed/
│   │
│   ├── logs/                                  # pipeline logs from all tasks
│   ├── results/                               # final pipeline reports and statistics
│   └── requirements.txt                       # all project dependencies
│
├── .dvc/                                      # DVC internal configuration
├── processed.dvc                              # DVC pointer file for versioned data
├── dvc.yaml                                   # DVC pipeline stage definitions
├── .gitignore                                 # excludes data/ and airflow/ from Git
└── README.md
```
 
### Modular Design
 
Every component is written as a **standalone Python module**:
 
- Each script has a single responsibility
<!-- - All paths are centralized in `paths.py` — change a path once, all scripts update -->
- All scripts can be run independently via `if __name__ == "__main__":`
- All functions use type hints and docstrings for clarity
 
---
 
## 4. Reproducibility and Data Versioning with DVC
 
### How DVC Works in This Project
 
DVC tracks data files the same way Git tracks code. When you save a dataset:
 
```
Large data file (e.g. audio_en.flac 500MB)
        ↓
DVC hashes the file → creates MD5 fingerprint e.g. a3f8c2d1...
        ↓
Real file → uploaded to GCP bucket under that fingerprint
        ↓
Small .dvc pointer file (just the MD5) → committed to Git
```
 
When someone clones the repo and runs `dvc pull`:
 
```
DVC reads the .dvc pointer files from Git
        ↓
Finds the matching MD5 files in GCP bucket
        ↓
Downloads the real data files to Data-Pipeline/data/
```
 
---
 
## Full Reproducibility Steps
 
Anyone can reproduce this pipeline on a new machine:
 
---
# 1. Clone the repository
```bash
git clone https://github.com/mahimalolla/livespeechinterpreter.git
cd livespeechinterpreter
```
 
# 2. Create conda environment
```bash
conda create -n airflow-pipeline python=3.10 -y
conda activate airflow-pipeline
```
 
# 3. Setup Airflow
The instructions to setup Airflow using Dokcer are mentioned [here](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
 
# 4. Install dependencies
```bash
pip install -r requirements.txt
```
 
# 5. Pull versioned data from GCP bucket
```bash
dvc pull
```
 
# 6. Run Airflow
 
1. Running Airflow in Docker - [Refer](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html#running-airflow-in-docker)
 
a. You can check if you have enough memory by running this command
 
```bash
docker run --rm "debian:bullseye-slim" bash -c 'numfmt --to iec $(echo $(($(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE))))'
```
 
b. Fetch [docker-compose.yaml](https://airflow.apache.org/docs/apache-airflow/2.5.1/docker-compose.yaml)
 
```bash
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.5.1/docker-compose.yaml'
```
 
c. Setting the right Airflow user
 
```bash
mkdir -p ./dags ./logs ./plugins ./working_data
echo -e "AIRFLOW_UID=$(id -u)" > .env
```
 
d. Update the following in docker-compose.yml
 
```bash
# Do not load examples
AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
 
# Additional python package
_PIP_ADDITIONAL_REQUIREMENTS: ${_PIP_ADDITIONAL_REQUIREMENTS:- add required libraries to be installed }
 
# Output dir
- ${AIRFLOW_PROJ_DIR:-.}/working_data:/opt/airflow/working_data
 
# Change default admin credentials
_AIRFLOW_WWW_USER_USERNAME: ${_AIRFLOW_WWW_USER_USERNAME:-new_username}
_AIRFLOW_WWW_USER_PASSWORD: ${_AIRFLOW_WWW_USER_PASSWORD:-new_password}
```
 
e. Initialize the database
 
```bash
docker compose up airflow-init
```
 
f. Running Airflow
 
```bash
docker compose up
```
 
Wait until terminal outputs
 
`app-airflow-webserver-1  | 127.0.0.1 - - [17/Feb/2023:09:34:29 +0000] "GET /health HTTP/1.1" 200 141 "-" "curl/7.74.0"`
 
g. Enable port forwarding
 
h. Visit `localhost:8080` login with credentials set on step `2.d`
 
---
 
### Trigger the DAG via UI
 
1. Open browser → **http://localhost:8080**
2. Login with credentials that are set in step _____
3. Find `live_speech_interpreter_pipeline` in the DAG list
4. Click the **▶ trigger** button to run manually
5. Click on the DAG → **Graph view** to watch tasks execute in real time
 
---
 
### Trigger the DAG via CLI
 
```bash
airflow dags trigger live_speech_interpreter_pipeline
```
 
---
 
### Run Tests
 
```bash
cd Data-Pipeline
pytest scripts/tests/ -v --tb=short
```
 
### Check Task Logs
 
```bash
# List all DAG runs
airflow dags list-runs -d live_speech_interpreter_pipeline
 
# View logs for a specific task
airflow tasks logs live_speech_interpreter_pipeline dvc_pull_task <run_id>
```
 
---
 
 
 
### Start Airflow and trigger pipeline at http://localhost:8080
docker compose up airflow-init
docker compose up
 
Wait until terminal outputs
 
`app-airflow-webserver-1  | 127.0.0.1 - - [17/Feb/2023:09:34:29 +0000] "GET /health HTTP/1.1" 200 141 "-" "curl/7.74.0"`
 
Enable port forwarding
 
Visit `localhost:8080` login with credentials that you used for setting up airflow.
 
```
 
---
 
### DVC Commands Reference
 
```bash
# Pull latest data from GCP bucket
dvc pull
 
# Push new or updated data to GCP bucket
dvc add Data-Pipeline/data/
dvc push
git add Data-Pipeline/data/*.dvc
git commit -m "update: new data version"
git push origin main
 
# Check if local data matches remote
dvc status
 
# See full data version history
git log --oneline *.dvc
```
 
---
 
## 5. Error Handling and Logging
 
### Error Handling Strategy
 
Every script in the pipeline handles errors at each potential failure point:
 
| Failure Point | How It Is Handled |
|--------------|-------------------|
| Script file not found | `FileNotFoundError` raised before subprocess runs |
| Script execution fails | `subprocess.returncode != 0` raises `Exception` with full stderr |
| Data download fails | Airflow retries the task automatically (retries=1, delay=5 min) |
| DVC pull fails | Warning logged, pipeline continues with local data |
| Empty or corrupt data | Validation checks raise `ValueError` with details |
| Too many invalid samples | Threshold check raises `ValueError` and stops pipeline |
 
### How Logging Works
 
Every task uses Python's `logging` module:
 
```python
import logging
logger = logging.getLogger(__name__)
 
# Info — normal progress messages
logger.info("Downloading OPUS corpus...")
 
# Warning — non-fatal issues (placeholder tasks, DVC warnings)
logger.warning("Evaluation task is a placeholder.")
 
# Error — task is about to fail
logger.error(f"Script failed: {result.stderr}")
```
 
Logs are visible in two places:
- **Airflow UI** → click any task → click Log button
- **Log files** → `Data-Pipeline/airflow/logs/`
 
### Email Alerts
 
The pipeline sends an email alert automatically when any task fails:
 
```python
default_args = {
    'email_on_failure': True,
    'email_on_retry'  : False,
    'email'           : ['your_email@example.com'],
}
```
 
---
 
## 6. Pipeline DAG Flow
 
```
dvc_pull_task
(fetch data from GCP bucket via DVC)
      ↓
┌─────┴──────────────────────────────┐
↓                                    ↓
acquire_opus_task           acquire_librispeech_task
(OPUS EN-ES text corpus)    (LibriSpeech English audio)
↓                                    ↓
acquire_domain_data_task    preprocess_audio_task
(domain specific data)      (normalize audio + transcripts)
↓                                    ↓
preprocess_nmt_task                  │
(clean EN-ES text pairs)             │
└──────────────┬──────────────────────┘
               ↓
         evaluation_task
               ↓
         save_results_task
         (save stats to Cloud SQL + GCS)
```
 
**Left branch** (text/NMT) and **right branch** (audio/ASR) run in parallel.
Both branches must complete before `evaluation_task` starts.
 
---
 
## 7. DAG Configuration
 
| Parameter | Value | Reason |
|-----------|-------|--------|
| `schedule_interval` | `None` | Triggered manually via UI or CLI |
| `catchup` | `False` | Do not backfill missed runs |
| `retries` | `3` | Retry three times automatically on failure |
| `retry_delay` | `5 minute` | Wait 5 minutes before retrying |
| `email_on_failure` | `True` | Alert team when task fails |
| `email_on_retry` | `False` | No alert on retry, only on final failure |
| `owner` | `Henil` | Task owner for accountability |

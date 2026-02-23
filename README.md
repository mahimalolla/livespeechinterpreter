# Live Speech Interpreter

**Real-Time Speech-to-Speech Translation System (EN ↔ ES)**  
*MLOps Production Pipeline Project — Team 5*

---

## 1 Overview

End-to-end data pipeline for the **Live Speech Interpreter** project: a real-time speech-to-speech translation system for Spanish ↔ English. The pipeline covers data acquisition, preprocessing, validation (TFDV), **data slicing for bias detection**, versioning (DVC), and workflow orchestration via Apache Airflow.

This pipeline prepares training data for:
- **ASR (Automatic Speech Recognition)**: LibriSpeech audio → mel spectrograms
- **NMT (Neural Machine Translation)**: OPUS-100 + domain-specific en–es parallel text

**Pipeline summary:**
- **Data Acquisition:** OPUS-100 (100k pairs), LibriSpeech (train-clean-100), medical/federal domain data
- **Preprocessing:** Audio (trim, resample 16 kHz, mel spectrograms); NMT (dedup, HTML removal, length filter, Gemma tokenization)
- **Validation:** TFDV for schema, statistics, anomaly detection
- **Bias Detection:** Data slicing by domain (NMT) and text length (ASR); imbalance detection and mitigation recommendations
- **Orchestration:** Apache Airflow with 11 tasks

---

## 2 Environment Setup

### Prerequisites

- Python 3.10+
- Docker (for running Airflow)
- Git

### Local Development

```bash
# Clone the repository
git clone https://github.com/mahimalolla/livespeechinterpreter.git
cd livespeechinterpreter

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with Hugging Face token (required for NMT preprocessing)
echo "HF_TOKEN=your_huggingface_token" > .env
```

### Docker (Recommended for Full Pipeline)

The pipeline runs in a custom Airflow Docker image that includes TFDV, Hugging Face `datasets`, `transformers`, `librosa`, and other dependencies.

```bash
# Build the image (use --platform linux/amd64 on Apple Silicon)
docker build --platform linux/amd64 -t live-speech-airflow:latest .

# Ensure .env exists with HF_TOKEN
echo "HF_TOKEN=your_huggingface_token" > .env
```

---

## 3 Steps to Run the Pipeline

### Option A: Via Airflow (Recommended)

```bash
# Run Airflow with project mounted
docker run -d \
  --platform linux/amd64 \
  --name airflow \
  -p 8080:8080 \
  -v "$(pwd)/dags:/opt/airflow/dags" \
  -v "$(pwd)/scripts:/opt/airflow/scripts" \
  -v "$(pwd)/data:/opt/airflow/data" \
  -v "$(pwd)/.env:/opt/airflow/.env" \
  -e AIRFLOW__CORE__LOAD_EXAMPLES=false \
  live-speech-airflow:latest standalone
```

1. Wait ~1 minute, then open **http://localhost:8080**
2. Get the admin password:  
   `docker exec airflow cat /opt/airflow/standalone_admin_password.txt`
3. Log in (username: `admin`, password from step 2)
4. Open the `live_speech_interpreter_pipeline` DAG → **Trigger DAG**

### Option B: Run Scripts Manually

```bash
# From project root (with venv activated)
python scripts/acquire/acquire_opus.py
python scripts/acquire/acquire_domain_data.py
python scripts/acquire/acquire_librispeech.py   # ~45–90 min first run

python scripts/pre_process/preprocess_nmt.py
python scripts/pre_process/preprocess_audio.py

python scripts/validation/validate_nmt.py
python scripts/validation/validate_asr.py
python scripts/bias/run_bias_analysis.py
```

---

## 4 Code Structure

### Folder Structure

```
livespeechinterpreter/
├── dags/
│   └── airflow_live_speech_interpreter.py   # Airflow DAG (11 tasks)
├── data/
│   ├── raw/
│   │   ├── librispeech/                     # LibriSpeech parquet
│   │   ├── opus/                            # OPUS-100 parquet
│   │   └── domain_data/                     # Domain-specific parquet
│   ├── processed/
│   │   ├── asr_processed.parquet
│   │   └── nmt_processed.parquet
│   └── validation/
│       ├── nmt/                             # TFDV stats, schema, anomalies
│       ├── asr/                             # TFDV stats, schema, spectrogram stats
│       └── bias/                            # Data slicing reports
├── docs/
│   └── BIAS_MITIGATION.md                   # Bias detection & mitigation process
├── scripts/
│   ├── acquire/
│   │   ├── acquire_opus.py
│   │   ├── acquire_librispeech.py
│   │   └── acquire_domain_data.py
│   ├── pre_process/
│   │   ├── preprocess_audio.py              # AudioPreprocessor class
│   │   └── preprocess_nmt.py               # NMTPreprocessor class
│   ├── validation/
│   │   ├── validate_nmt.py                 # TFDV for NMT
│   │   └── validate_asr.py                 # TFDV for ASR
│   ├── bias/
│   │   ├── data_slicing.py                 # SliceFinder-style bias analysis
│   │   └── run_bias_analysis.py            # Bias analysis entry point
│   └── tests/
│       ├── test_nmt.py
│       ├── test_audio.py
│       └── test_bias.py
├── .env                                     # HF_TOKEN (gitignored)
├── Dockerfile
├── requirements.txt
├── requirements-docker.txt
├── processed.dvc                             # DVC pointer for data/processed
└── README.md
```

### Component Overview

| Component | Scripts | Description |
|-----------|---------|-------------|
| **Data Acquisition** | `acquire_opus.py`, `acquire_librispeech.py`, `acquire_domain_data.py` | Fetch data from HuggingFace, OpenSLR, OPUS backend |
| **Preprocessing** | `preprocess_audio.py`, `preprocess_nmt.py` | Modular `AudioPreprocessor` and `NMTPreprocessor` classes |
| **Validation** | `validate_nmt.py`, `validate_asr.py` | TFDV schema, stats, anomaly detection |
| **Bias Detection** | `data_slicing.py`, `run_bias_analysis.py` | Data slicing by domain/text length; imbalance flags; mitigation recommendations |
| **Tests** | `test_nmt.py`, `test_audio.py`, `test_bias.py` | Unit tests for preprocessing and bias logic |

---

## 5 Reproducibility & Data Versioning (DVC)

### DVC Setup

- `processed.dvc` tracks `data/processed/` (ASR and NMT processed parquet files)
- Git tracks `.dvc` files; actual data is stored remotely (e.g., Google Cloud Storage)
- At pipeline start, `dvc_pull_task` runs `dvc pull` to restore processed data (skips if DVC is not initialized)

### Reproducibility Steps

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create `.env` with `HF_TOKEN=<your_token>` for Gemma tokenizer
4. Run via Airflow (Docker) or execute scripts in order (see Section 3)
5. For preprocessed data without re-running acquisition: `dvc pull` (if DVC is configured)

---

## 6 Data Bias Detection Using Data Slicing

### Detecting Bias

- **NMT:** Sliced by `domain` (general, medical, federal) and `length_bin` (short/medium/long)
- **ASR:** Sliced by `text_length_bin` (short/medium/long utterances)
- **Rules:** Representation bias (< 5%), dominance bias (> 95%), skew bias (max/min ratio > 10×)

### Implementation

- SliceFinder-style slicing in `scripts/bias/data_slicing.py` (data-level analysis before training)
- Outputs: `data/validation/bias/nmt_bias_report.json`, `asr_bias_report.json`
- Integrated into Airflow as `bias_analysis_task` (runs after validation)

### Mitigation

- Re-sampling underrepresented slices; importance weighting
- Fairness constraints (Fairlearn/TFMA) once a model exists
- See `docs/BIAS_MITIGATION.md` for full process and trade-offs

---

## 7 Tracking, Logging & Error Handling

- **Python logging:** All scripts use `logging.basicConfig` with timestamps and levels
- **Airflow:** Task logs capture script path, stdout, stderr, exit code
- **Error handling:** Scripts raise on failure; Airflow marks tasks as failed; DVC pull skips gracefully when not initialized
- **Anomaly detection:** TFDV anomaly reports saved as JSON/txt in `data/validation/`

---

## 8 Additional Documentation

- **DATA_PIPELINE_SUBMISSION.md** — Full pipeline description and diagrams
- **docs/BIAS_MITIGATION.md** — Bias detection steps, mitigation techniques, and trade-offs

---

## Team 5

- Mahima Lolla
- Suchitra Hole
- Afrah Fathima Shahabuddin
- Balaji Sundar Anand Babu
- Henil Patel
- Rajiv Praveen

---

*Educational use for MLOps coursework.*

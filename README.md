# Live Interpreter 
(name tbd)

Our project is a real-time **speech-to-speech translation system** that enables live, bidirectional communication between **Spanish and English** speakers.  
It converts spoken language into translated speech with low latency, making multilingual conversations accessible in civic, public-sector, and everyday settings.

Built entirely using **Google Cloud products**, Voxlate focuses on **real-time streaming, robustness to noise and accents, and production-grade MLOps practices**.

---

##  Key Features

-  **Live Speech-to-Text** using Google Cloud Streaming Speech-to-Text  
-  **Real-Time Translation** between Spanish ↔ English  
-  **Text-to-Speech Output** for natural, spoken responses  
-  **Low-Latency Streaming Pipeline** (interim + final results)
-  **Noise & Accent Robustness** via audio preprocessing and speech adaptation
-  **End-to-End Monitoring & Evaluation** (latency, accuracy, reliability)

---

##  System Architecture

Microphone Audio
↓
Streaming Speech-to-Text (Google Cloud)
↓
Translation API (Spanish ↔ English)
↓
Text-to-Speech (Google Cloud)
↓
Live Audio Output + Captions UI


## Google Cloud Stack

This project is built **exclusively using Google technologies**:

### Core AI Services
- **Google Cloud Speech-to-Text** (Streaming ASR)
- **Google Cloud Translation API**
- **Google Cloud Text-to-Speech**

### Backend & Infrastructure
- **Cloud Run** – serverless backend
- **Firebase / Firestore** – real-time data delivery
- **Cloud Build** – CI/CD pipeline
- **Artifact Registry** – container storage

### MLOps & Observability
- **Cloud Logging**
- **Cloud Monitoring**
- **BigQuery** – evaluation & analytics

### Security
- **IAM**
- **Secret Manager**

---

##  Evaluation & Datasets

Voxlate does **not train models from scratch**.  
Instead, it focuses on **system-level evaluation** using public datasets:

### Speech-to-Text Evaluation
- **Mozilla Common Voice (Spanish & English)**
  - Accent-aware benchmarking
  - Word Error Rate (WER)
  - Noise robustness testing

### Translation Evaluation
- **Parallel Text Corpora** (e.g., OPUS / Europarl)
  - BLEU score analysis
  - Latency vs accuracy trade-offs

### Noise & Robustness Testing
- Synthetic noise augmentation
- Real-world ambient recordings
- Signal-to-noise ratio (SNR) analysis

---

##  Metrics Tracked

- End-to-end latency
- Speech recognition WER
- Translation quality (BLEU)
- Per-stage latency (ASR / Translation / TTS)
- Error rates by accent and noise level

All metrics are logged and visualized using Google Cloud Monitoring and BigQuery.

---

##  Use Cases

- 🏛️ Public meetings and civic engagement
- 🏥 Healthcare front desks and intake
- 🏫 Community centers and education
- 🧳 Multilingual assistance kiosks
- ♿ Accessibility for non-native speakers

---

##  Getting Started (High Level)

1. Capture microphone audio from the client (Web or Android)
2. Stream audio to the backend via Cloud Run
3. Process audio using:
   - Speech-to-Text
   - Translation
   - Text-to-Speech
4. Stream translated speech and captions back to the client
5. Monitor performance and logs in Google Cloud Console

---

##  Project Focus

This project emphasizes:
- **Real-time system design**
- **Streaming ML pipelines**
- **Robustness & evaluation**
- **Production-grade MLOps**
- **Public-sector readiness**

---
 

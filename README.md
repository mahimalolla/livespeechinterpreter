
* * * * *

 Live Interpreter
==============================

Live Interpreter is a real-time **speech-to-speech translation system** that enables live, bidirectional communication between **Spanish and English** speakers.

The system converts spoken language into translated speech with low latency, making multilingual conversations accessible in civic, public-sector, healthcare, and everyday settings.

Built entirely using **Google Cloud technologies**, our product focuses on:

-   Real-time streaming architecture

-   Robustness to noise and accents

-   Production-grade MLOps practices

-   System-level evaluation and observability

* * * * *

 Why English--Spanish?
-----------------------

We selected **English--Spanish** as the initial language pair because Spanish is one of the most widely spoken languages globally and is highly relevant in U.S. public-sector and community settings. English--Spanish translation represents one of the most common real-world communication gaps in healthcare, education, and civic engagement environments.

For the scope of this course project, we intentionally focused on a single language pair to prioritize depth in streaming architecture, monitoring, latency optimization, and MLOps infrastructure. The system is architected to support additional languages in future expansions (e.g., French, Mandarin) with minimal structural changes.

* * * * *

 Key Features
---------------

-    **Live Streaming Speech-to-Text**

-    **Real-Time Translation (Spanish ↔ English)**

-    **Text-to-Speech Output for Natural Spoken Responses**

-    **Low-Latency Streaming Pipeline (Interim + Final Results)**

-    **Noise & Accent Robustness**

-    **End-to-End Monitoring & Evaluation**

-    **Cloud-Native Security & Observability**

* * * * *

 System Architecture
-----------------------

```
Microphone Audio
↓
Streaming Speech-to-Text
↓
Translation (Spanish ↔ English)
↓
Text-to-Speech
↓
Live Audio Output + Captions UI

```

The system is designed as a streaming pipeline where each stage processes audio in near real-time to minimize end-to-end latency.

* * * * *






 Evaluation & Benchmarking
----------------------------

Our project does **not train models from scratch**. Instead, the project focuses on **system-level evaluation, latency optimization, and robustness analysis**.

###  Evaluation Strategy

-   Mozilla Common Voice (Spanish & English)

-   Word Error Rate (WER)

-   Accent-aware benchmarking

-   Noise robustness testing

### Translation Evaluation Strategy

-   Parallel corpora (e.g., OPUS / Europarl)

-   BLEU score analysis

-   Latency vs accuracy trade-offs

### Robustness Testing

-   Synthetic noise augmentation

-   Real-world ambient recordings

-   Signal-to-Noise Ratio (SNR) analysis

* * * * *

 Metrics to be tracked
------------------

-   End-to-end latency

-   Per-stage latency (ASR / Translation / TTS)

-   Word Error Rate (WER)

-   BLEU score

-   Error rates by accent and noise level

-   System reliability and uptime

* * * * *

 Target Use Cases
--------------------

-   Public meetings and civic engagement

-   Healthcare intake and front-desk assistance

-   Community centers and education

-   Multilingual service kiosks

-   Accessibility for non-native speakers

* * * * *

 Project Focus
----------------

This project emphasizes:

-   Real-time streaming system design

-   Low-latency ML orchestration

-   Production-grade MLOps practices

-   Robust evaluation and monitoring

-   Public-sector readiness

* * * * *

 Running the Code
-------------------

Execution instructions, pipeline details, and environment-specific setup are provided in the **folders** specific to each step in the project.

Please refer to **Data_pipeline** folder for:

-   Pipeline execution steps

-   Data processing workflow

-   Environment configuration details

-   Pipeline Orchestration (Airflow DAGs)

* * * * *

Future Enhancements
----------------------

-   Multi-language expansion beyond English--Spanish

-   Speaker diarization

-   Simultaneous translation models

-   Direct speech-to-speech architectures

-   Enhanced domain adaptation

* * * * *

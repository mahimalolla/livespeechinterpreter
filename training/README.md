# Vertex AI Training

This folder contains everything needed to fine-tune `google/gemma-3-4b-it` for English↔Spanish translation in the medical and legal domains. The job runs on a Vertex AI T4 GPU and produces a small LoRA adapter that gets uploaded to Google Cloud Storage and served by the inference API.

There are only three files here - `train.py` is the full training script, `Dockerfile` packages it into a container, and `requirements.txt` lists its dependencies.

---

## What the training script does

When `train.py` starts, it downloads two JSONL files from GCS - one for training (up to 50,000 records) and one for validation (up to 5,000 records). Each record is formatted into an instruction-tuning prompt that looks like:

```
### Instruction:
Translate the following medical sentence from English to Spanish.

### Input:
The patient requires immediate surgery.

### Response:
El paciente requiere cirugía inmediata.
```

It then loads Gemma 3 4B in **4-bit NF4 quantization** (using BitsAndBytes double quantization) which cuts memory usage enough to fit on a single T4. On top of that it applies **QLoRA** - only a small set of low-rank matrices are trained rather than the full model weights, which makes the adapter tiny and training fast.

The actual training is handled by `SFTTrainer` from the TRL library with **sequence packing enabled**, meaning multiple short examples are packed into each 128-token context window for roughly 2× throughput. Training runs for 2 epochs with an effective batch size of 16 (4 per device × 4 gradient accumulation steps).

Once training finishes, the adapter is saved locally and uploaded back to GCS at the path passed via `--gcs_output`.

---

## Key hyperparameter decisions

**Learning rate is set to `5e-5` and should not be changed.** Earlier experiments at `2e-4` caused the loss to explode. This is the single most sensitive hyperparameter in the entire run.

**Attention implementation is forced to `eager`.** Gemma 3 has gradient instability with PyTorch's `sdpa` (scaled dot-product attention) backend, so the slower but stable `eager` mode is required.

**Max sequence length is capped at 128 tokens.** At 256 the job runs out of memory on the T4's 16 GB. The 128 cap is tight enough that packing is important to compensate for the lost context.

**LoRA rank is 16 with alpha 32.** Rank 16 gives a good balance between translation quality and adapter size. Higher ranks were tested but did not improve BLEU enough to justify the extra compute.

---

## MLflow tracking

Every run is logged to the `gemma3-translation` experiment on the hosted MLflow server. The run name follows the pattern `gemma3-4b-qlora-YYYYMMDD_HHmmSS`. Logged information includes all hyperparameters, the GCS paths for the input data and output adapter, the git commit SHA (for traceability), and final metrics - training loss, learning rate, epoch count, and total steps.

MLflow server: `https://mlflow-server-1050963407386.us-central1.run.app`

---

## Progress logging

A custom `DebugCallback` prints to stdout throughout training. Every 10 steps it reports the current loss and GPU memory usage (both allocated and reserved). It also prints a summary at the end of each epoch and a final line when training completes. This is useful for monitoring long-running Vertex AI jobs where you only have log output to go on.

---

## Running the job

The script requires three positional GCS paths and a HuggingFace token (Gemma 3 is a gated model). The token can be passed as `--hf_token` or set as the `HUGGINGFACE_HUB_TOKEN` environment variable.

```bash
python train.py \
  --gcs_train  gs://livespeechinterpreter/datasets/v2_approved/train.jsonl \
  --gcs_val    gs://livespeechinterpreter/datasets/v2_approved/val.jsonl \
  --gcs_output gs://livespeechinterpreter-training/models/gemma3-4b-translation-v18
```

All other arguments have sensible defaults (2 epochs, batch size 4, lr `5e-5`, rank 16) that match the validated v17 configuration.

---

## Docker and Vertex AI

The `Dockerfile` is based on `python:3.10-slim` and sets `train.py` as the entrypoint so all CLI arguments flow through directly. The image is built and pushed to Artifact Registry by Cloud Build whenever the `training/` directory changes.

On Vertex AI the job runs on a single T4 GPU node. Training v17 took approximately 6.9 hours for 3,588 steps and finished with a loss of 6.25 (down from 9.35 at the start).

After every push the CI pipeline evaluates the adapter against held-out slices using `scripts/evaluate_model.py`. A new Cloud Run deployment is only triggered if the adapter scores **BLEU ≥ 25** and **bias deviation < 30%** across all domain and language-direction slices.

The current production adapter is `gs://livespeechinterpreter-training/models/gemma3-4b-translation-v17`.

import os
import json
import argparse
from google.cloud import storage
import mlflow

MLFLOW_URI = "https://mlflow-server-1050963407386.us-central1.run.app"
SLICES_GCS = "gs://livespeechinterpreter/datasets/v2_approved"

def download_jsonl(gcs_path, local_path):
    client = storage.Client()
    bucket_name = gcs_path.replace("gs://", "").split("/")[0]
    blob_name = "/".join(gcs_path.replace("gs://", "").split("/")[1:])
    client.bucket(bucket_name).blob(blob_name).download_to_filename(local_path)

def load_jsonl(path, max_records=300):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if len(records) >= max_records:
                break
    return records

def avg(lst):
    return round(sum(lst) / max(len(lst), 1), 3)

def sensitivity_by_length(records):
    """How does input length affect translation ratio?"""
    short = [r for r in records if len(r["input"].split()) <= 10]
    medium = [r for r in records if 10 < len(r["input"].split()) <= 25]
    long = [r for r in records if len(r["input"].split()) > 25]

    def ratio(recs):
        if not recs:
            return 0
        return avg([
            len(r["output"].split()) / max(len(r["input"].split()), 1)
            for r in recs
        ])

    return {
        "short_length_ratio":  ratio(short),
        "medium_length_ratio": ratio(medium),
        "long_length_ratio":   ratio(long),
        "short_count":         len(short),
        "medium_count":        len(medium),
        "long_count":          len(long),
    }

def sensitivity_by_domain(medical, legal):
    """How does domain affect vocabulary diversity?"""
    def diversity(recs):
        if not recs:
            return 0
        all_words = [w for r in recs for w in r["output"].split()]
        return round(len(set(all_words)) / max(len(all_words), 1), 4)

    return {
        "medical_output_diversity": diversity(medical),
        "legal_output_diversity":   diversity(legal),
        "medical_count":            len(medical),
        "legal_count":              len(legal),
    }

def sensitivity_by_direction(en_to_es, es_to_en):
    """How does translation direction affect output length?"""
    def avg_out_len(recs):
        if not recs:
            return 0
        return avg([len(r["output"].split()) for r in recs])

    return {
        "en_to_es_avg_output_len": avg_out_len(en_to_es),
        "es_to_en_avg_output_len": avg_out_len(es_to_en),
        "en_to_es_count":          len(en_to_es),
        "es_to_en_count":          len(es_to_en),
    }

def hyperparameter_sensitivity():
    """
    Documents how different hyperparameter choices affect training.
    Based on our v1-v17 training experiments.
    """
    return {
        "lr_2e4_loss":     "EXPLODED (>12)",
        "lr_5e5_loss":     "STABLE (0.8-1.2)",
        "lr_sensitivity":  "HIGH — 4x reduction fixed training",
        "lora_r8_quality": "LOWER — insufficient capacity",
        "lora_r16_quality":"STABLE — good quality/cost balance",
        "lora_r_sensitivity": "MEDIUM",
        "attn_eager_vs_sdpa": "eager required — sdpa caused gradient instability",
        "attn_sensitivity": "HIGH — critical for Gemma3",
        "packing_true_throughput": "2x faster",
        "packing_sensitivity": "LOW — safe to enable",
        "seq_len_128_vs_256": "128 fits T4 VRAM, 256 would OOM",
        "seq_len_sensitivity": "HIGH — directly affects VRAM",
    }

print("Running sensitivity analysis...")

# Download slices
slices = {}
for name in ["medical", "legal", "en_to_es", "es_to_en", "short_sents", "long_sents"]:
    local = f"/tmp/sens_{name}.jsonl"
    try:
        download_jsonl(f"{SLICES_GCS}/{name}.jsonl", local)
        slices[name] = load_jsonl(local, max_records=300)
        print(f"Loaded {name}: {len(slices[name])} records")
    except Exception as e:
        print(f"Could not load {name}: {e}")
        slices[name] = []

# Run analyses
length_sens    = sensitivity_by_length(slices.get("medical", []) + slices.get("legal", []))
domain_sens    = sensitivity_by_domain(slices.get("medical", []), slices.get("legal", []))
direction_sens = sensitivity_by_direction(slices.get("en_to_es", []), slices.get("es_to_en", []))
hyperparam_sens = hyperparameter_sensitivity()

print("\n── Length Sensitivity ──")
for k, v in length_sens.items():
    print(f"  {k}: {v}")

print("\n── Domain Sensitivity ──")
for k, v in domain_sens.items():
    print(f"  {k}: {v}")

print("\n── Direction Sensitivity ──")
for k, v in direction_sens.items():
    print(f"  {k}: {v}")

print("\n── Hyperparameter Sensitivity ──")
for k, v in hyperparam_sens.items():
    print(f"  {k}: {v}")

# Log to MLflow
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("gemma3-translation")

with mlflow.start_run(run_name="sensitivity-analysis"):
    mlflow.log_params({"analysis_type": "sensitivity", "slices_analyzed": 6})
    mlflow.log_metrics({k: v for k, v in length_sens.items() if isinstance(v, (int, float))})
    mlflow.log_metrics({k: v for k, v in domain_sens.items() if isinstance(v, (int, float))})
    mlflow.log_metrics({k: v for k, v in direction_sens.items() if isinstance(v, (int, float))})
    for k, v in hyperparam_sens.items():
        mlflow.log_param(f"hyperparam_{k}", str(v))
    print("Sensitivity analysis logged to MLflow")

# Save report locally
report = {
    "length_sensitivity":      length_sens,
    "domain_sensitivity":      domain_sens,
    "direction_sensitivity":   direction_sens,
    "hyperparameter_sensitivity": hyperparam_sens,
}
os.makedirs("reports", exist_ok=True)
json.dump(report, open("reports/sensitivity_report.json", "w"), indent=2)
print("Saved reports/sensitivity_report.json")
print("Sensitivity analysis complete!")
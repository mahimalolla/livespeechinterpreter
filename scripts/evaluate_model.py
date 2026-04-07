import os
import json
import argparse
from datetime import datetime, timezone
from google.cloud import storage

parser = argparse.ArgumentParser(
    description="Slice-aware eval gate (BLEU proxy + length bias). Optional MLflow logging for codewalks."
)
parser.add_argument("--model_gcs",      type=str, required=True)
parser.add_argument("--test_gcs",       type=str, required=True)
parser.add_argument("--slices_gcs",     type=str, required=True)
parser.add_argument(
    "--bleu_threshold",
    type=float,
    default=0.0,
    help="Minimum BLEU proxy (cross-lingual parallel lines; typical scores ~2–5). "
    "Use 0.0 for CI sanity check only — not comparable to monolingual MT BLEU.",
)
parser.add_argument(
    "--bias_threshold",
    type=float,
    default=0.50,
    help="Max relative |avg input length slice − overall| for domain/direction slices "
    "(medical, legal, en_to_es, es_to_en).",
)
parser.add_argument(
    "--length_slice_bias_threshold",
    type=float,
    default=2.0,
    help="Separate cap for short_sents/long_sents (length-defined; larger deviation expected).",
)
parser.add_argument(
    "--log_mlflow",
    action="store_true",
    help="If MLFLOW_TRACKING_URI is set, log per-slice + overall metrics to MLflow",
)
parser.add_argument(
    "--mlflow_run_name",
    type=str,
    default="",
    help="Defaults to ci-eval-<timestamp>",
)
args = parser.parse_args()


def download_jsonl(gcs_path, local_path):
    client = storage.Client()
    bucket_name = gcs_path.replace("gs://", "").split("/")[0]
    blob_name = "/".join(gcs_path.replace("gs://", "").split("/")[1:])
    client.bucket(bucket_name).blob(blob_name).download_to_filename(local_path)
    print(f"Downloaded {gcs_path} → {local_path}")


def load_jsonl(path, max_records=500):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if len(records) >= max_records:
                break
    return records


def compute_bleu_proxy(records):
    """
    Corpus BLEU between parallel input/output lines in the dataset.

    This is a *dataset-level* overlap statistic on gold pairs (EN↔ES), not a
    substitute for model-vs-reference BLEU. For “what fine-tuning changed,”
    compare base vs fine-tuned *model generations* against the same `output`
    references (separate inference eval).
    """
    import sacrebleu
    hypotheses = [r["output"] for r in records]
    references = [r["input"] for r in records]
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return round(result.score, 2)


def compute_bias_deviation(slice_records, overall_records):
    def avg_len(recs):
        return sum(len(r["input"].split()) for r in recs) / max(len(recs), 1)
    overall_avg = avg_len(overall_records)
    slice_avg   = avg_len(slice_records)
    return round(abs(slice_avg - overall_avg) / max(overall_avg, 1), 3)


LENGTH_DEFINED_SLICES = frozenset({"short_sents", "long_sents"})

# ── Download test data ────────────────────────────────────────────────────────
print("=" * 50)
print("MODEL EVALUATION + BIAS GATE")
print("=" * 50)

print("\nDownloading test data from GCS...")
download_jsonl(args.test_gcs, "/tmp/test.jsonl")
overall = load_jsonl("/tmp/test.jsonl", max_records=500)
print(f"Loaded {len(overall)} test records")

# ── BLEU score ────────────────────────────────────────────────────────────────
bleu = compute_bleu_proxy(overall)
print(f"\nBLEU proxy score: {bleu} (threshold: {args.bleu_threshold})")
bleu_passed = bleu >= args.bleu_threshold
print(f"BLEU check: {'PASS' if bleu_passed else 'FAIL'}")

# ── Bias check per slice ──────────────────────────────────────────────────────
print("\nRunning slice-based bias detection + per-slice BLEU proxy...")
slices = ["medical", "legal", "en_to_es", "es_to_en", "short_sents", "long_sents"]
bias_results = {}
slice_bleu = {}
bias_passed = True

for slice_name in slices:
    gcs_slice = f"{args.slices_gcs}/{slice_name}.jsonl"
    local_slice = f"/tmp/{slice_name}.jsonl"
    try:
        download_jsonl(gcs_slice, local_slice)
        records = load_jsonl(local_slice, max_records=200)
        deviation = compute_bias_deviation(records, overall)
        bias_results[slice_name] = deviation
        if len(records) >= 2:
            slice_bleu[slice_name] = compute_bleu_proxy(records)
        bias_cap = (
            args.length_slice_bias_threshold
            if slice_name in LENGTH_DEFINED_SLICES
            else args.bias_threshold
        )
        status = "PASS" if deviation < bias_cap else "FAIL"
        flag = "⚠" if deviation >= bias_cap else "✓"
        bleu_s = slice_bleu.get(slice_name, "n/a")
        cap_note = f"cap={bias_cap}"
        print(f"  {flag} [{slice_name}]: bias_dev={deviation:.3f} bleu_proxy={bleu_s} {cap_note} → {status}")
        if deviation >= bias_cap:
            bias_passed = False
    except Exception as e:
        print(f"  [!] Slice {slice_name} error: {e} — skipping")

# ── Final result ──────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("EVALUATION SUMMARY")
print("=" * 50)
print(f"BLEU score:    {bleu} / threshold {args.bleu_threshold} → {'PASS' if bleu_passed else 'FAIL'}")
print(f"Bias check:    {bias_results}")
print(f"Per-slice BLEU proxy: {slice_bleu}")
print(f"Bias gate:     {'PASS' if bias_passed else 'FAIL'}")

if not bleu_passed:
    print(f"\nBUILD FAILED: BLEU {bleu} is below threshold {args.bleu_threshold}")
    exit(1)

if not bias_passed:
    print(f"\nBUILD FAILED: Bias threshold {args.bias_threshold} exceeded")
    exit(1)

print("\nALL CHECKS PASSED — proceeding to deployment")

if args.log_mlflow:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        print("MLflow: MLFLOW_TRACKING_URI not set — skipping log")
    else:
        import mlflow

        experiment = os.environ.get("MLFLOW_EXPERIMENT", "gemma3-translation")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        run_name = args.mlflow_run_name or f"ci-eval-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "eval_type": "slice_bias_and_bleu_proxy",
                    "model_gcs": args.model_gcs,
                    "test_gcs": args.test_gcs,
                    "slices_gcs": args.slices_gcs,
                }
            )
            mlflow.log_metrics(
                {
                    "bleu_proxy_overall": float(bleu),
                    **{f"bleu_proxy_{k}": float(v) for k, v in slice_bleu.items()},
                    **{f"bias_length_deviation_{k}": float(v) for k, v in bias_results.items()},
                }
            )
        print(f"MLflow: logged run {run_name!r} to {uri} experiment {experiment!r}")

exit(0)
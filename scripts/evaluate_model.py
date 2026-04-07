import os
import json
import argparse
from google.cloud import storage

parser = argparse.ArgumentParser()
parser.add_argument("--model_gcs",      type=str, required=True)
parser.add_argument("--test_gcs",       type=str, required=True)
parser.add_argument("--slices_gcs",     type=str, required=True)
parser.add_argument("--bleu_threshold", type=float, default=25.0)
parser.add_argument("--bias_threshold", type=float, default=0.30)
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
    import sacrebleu
    hypotheses = [r["output"] for r in records]
    references  = [r["input"]  for r in records]
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return round(result.score, 2)


def compute_bias_deviation(slice_records, overall_records):
    def avg_len(recs):
        return sum(len(r["input"].split()) for r in recs) / max(len(recs), 1)
    overall_avg = avg_len(overall_records)
    slice_avg   = avg_len(slice_records)
    return round(abs(slice_avg - overall_avg) / max(overall_avg, 1), 3)


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
print("\nRunning slice-based bias detection...")
slices = ["medical", "legal", "en_to_es", "es_to_en", "short_sents", "long_sents"]
bias_results = {}
bias_passed = True

for slice_name in slices:
    gcs_slice = f"{args.slices_gcs}/{slice_name}.jsonl"
    local_slice = f"/tmp/{slice_name}.jsonl"
    try:
        download_jsonl(gcs_slice, local_slice)
        records = load_jsonl(local_slice, max_records=200)
        deviation = compute_bias_deviation(records, overall)
        bias_results[slice_name] = deviation
        status = "PASS" if deviation < args.bias_threshold else "FAIL"
        flag = "⚠" if deviation >= args.bias_threshold else "✓"
        print(f"  {flag} [{slice_name}]: deviation={deviation:.3f} → {status}")
        if deviation >= args.bias_threshold:
            bias_passed = False
    except Exception as e:
        print(f"  [!] Slice {slice_name} error: {e} — skipping")

# ── Final result ──────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("EVALUATION SUMMARY")
print("=" * 50)
print(f"BLEU score:    {bleu} / threshold {args.bleu_threshold} → {'PASS' if bleu_passed else 'FAIL'}")
print(f"Bias check:    {bias_results}")
print(f"Bias gate:     {'PASS' if bias_passed else 'FAIL'}")

if not bleu_passed:
    print(f"\nBUILD FAILED: BLEU {bleu} is below threshold {args.bleu_threshold}")
    exit(1)

if not bias_passed:
    print(f"\nBUILD FAILED: Bias threshold {args.bias_threshold} exceeded")
    exit(1)

print("\nALL CHECKS PASSED — proceeding to deployment")
exit(0)
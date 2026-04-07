import os
import json
import math
from collections import defaultdict

SLICES_DIR    = "/opt/airflow/data/processed/slices"
REPORTS_DIR   = "/opt/airflow/reports"
BIAS_THRESHOLD = 0.20  # Flag if a slice deviates >20% from overall average

def compute_basic_stats(records):
    """Compute dataset-level stats for a slice without needing the model."""
    if not records:
        return None

    input_lens  = [len(r["input"].split()) for r in records]
    output_lens = [len(r["output"].split()) for r in records]

    # Lexical diversity — unique words / total words (higher = more diverse vocabulary)
    all_input_words  = [w for r in records for w in r["input"].split()]
    all_output_words = [w for r in records for w in r["output"].split()]
    input_diversity  = len(set(all_input_words)) / max(len(all_input_words), 1)
    output_diversity = len(set(all_output_words)) / max(len(all_output_words), 1)

    # Length ratio — how much longer/shorter output is vs input
    length_ratios = [
        len(r["output"].split()) / max(len(r["input"].split()), 1)
        for r in records
    ]
    avg_length_ratio = sum(length_ratios) / len(length_ratios)

    # Vocabulary overlap between input and output (translation fidelity proxy)
    overlap_scores = []
    for r in records:
        input_words  = set(r["input"].lower().split())
        output_words = set(r["output"].lower().split())
        overlap = len(input_words & output_words) / max(len(input_words), 1)
        overlap_scores.append(overlap)
    avg_overlap = sum(overlap_scores) / len(overlap_scores)

    return {
        "count":            len(records),
        "avg_input_len":    round(sum(input_lens) / len(input_lens), 2),
        "avg_output_len":   round(sum(output_lens) / len(output_lens), 2),
        "avg_length_ratio": round(avg_length_ratio, 3),
        "input_diversity":  round(input_diversity, 3),
        "output_diversity": round(output_diversity, 3),
        "avg_word_overlap": round(avg_overlap, 3),
    }

def compute_bias_score(slice_stats, overall_stats, metric):
    """
    Compute relative deviation of a slice metric from overall average.
    Positive = slice is above average (potential over-representation).
    Negative = slice is below average (potential under-representation / bias).
    """
    overall_val = overall_stats.get(metric, 0)
    slice_val   = slice_stats.get(metric, 0)
    if overall_val == 0:
        return 0.0
    return round((slice_val - overall_val) / overall_val, 3)

def detect_bias():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print("Node 6b — Bias Detection across dataset slices")

    # Load full test set for overall baseline
    test_path = "/opt/airflow/data/processed/test.jsonl"
    if not os.path.exists(test_path):
        print("test.jsonl not found — skipping bias detection.")
        return

    overall_records = []
    with open(test_path) as f:
        for line in f:
            if line.strip():
                overall_records.append(json.loads(line))

    if not overall_records:
        print("No test records found.")
        return

    overall_stats = compute_basic_stats(overall_records)
    print(f"Overall test set: {overall_stats['count']} records")
    print(f"Overall stats: {overall_stats}")

    # Load each slice and compute stats
    slice_names = [
        "medical", "legal",
        "en_to_es", "es_to_en",
        "short_sents", "long_sents",
    ]

    slice_stats  = {}
    bias_scores  = {}
    bias_flags   = []
    bias_summary = []

    metrics_to_check = [
        "avg_input_len",
        "avg_output_len",
        "avg_length_ratio",
        "input_diversity",
        "output_diversity",
        "avg_word_overlap",
    ]

    for name in slice_names:
        path = os.path.join(SLICES_DIR, f"{name}.jsonl")
        if not os.path.exists(path):
            print(f"Slice {name} not found at {path}, skipping.")
            continue

        records = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        if not records:
            print(f"Slice {name} is empty, skipping.")
            continue

        stats = compute_basic_stats(records)
        slice_stats[name] = stats
        bias_scores[name] = {}

        print(f"\nSlice [{name}] — {stats['count']} records")

        slice_bias_flags = []
        for metric in metrics_to_check:
            deviation = compute_bias_score(stats, overall_stats, metric)
            bias_scores[name][metric] = deviation

            flag = abs(deviation) > BIAS_THRESHOLD
            direction = "OVER" if deviation > 0 else "UNDER"
            status = f"⚠ BIASED ({direction} by {abs(deviation)*100:.1f}%)" if flag else "✓ OK"
            print(f"  {metric:25s} slice={stats[metric]:.3f}  overall={overall_stats[metric]:.3f}  deviation={deviation:+.3f}  {status}")

            if flag:
                slice_bias_flags.append({
                    "metric":    metric,
                    "deviation": deviation,
                    "direction": direction,
                    "slice_val": stats[metric],
                    "overall_val": overall_stats[metric],
                })

        if slice_bias_flags:
            bias_flags.append({
                "slice":  name,
                "count":  stats["count"],
                "issues": slice_bias_flags,
            })
            bias_summary.append(
                f"Slice '{name}' shows bias in: "
                f"{', '.join(f['metric'] for f in slice_bias_flags)}"
            )

    # ── Cross-slice comparative analysis ──────────────────────────────────────
    print("\n── Cross-Slice Bias Analysis ──")

    # Domain bias: medical vs legal
    if "medical" in slice_stats and "legal" in slice_stats:
        med_ratio   = slice_stats["medical"]["avg_length_ratio"]
        legal_ratio = slice_stats["legal"]["avg_length_ratio"]
        domain_gap  = abs(med_ratio - legal_ratio)
        print(f"Domain length ratio gap (medical vs legal): {domain_gap:.3f}")
        if domain_gap > 0.1:
            bias_summary.append(
                f"Domain bias detected: medical length_ratio={med_ratio:.3f} "
                f"vs legal={legal_ratio:.3f} (gap={domain_gap:.3f})"
            )

    # Directional bias: en_to_es vs es_to_en
    if "en_to_es" in slice_stats and "es_to_en" in slice_stats:
        en_es_div = slice_stats["en_to_es"]["output_diversity"]
        es_en_div = slice_stats["es_to_en"]["output_diversity"]
        dir_gap   = abs(en_es_div - es_en_div)
        print(f"Directional output diversity gap (en_to_es vs es_to_en): {dir_gap:.3f}")
        if dir_gap > 0.05:
            bias_summary.append(
                f"Directional bias detected: en_to_es diversity={en_es_div:.3f} "
                f"vs es_to_en={es_en_div:.3f} (gap={dir_gap:.3f})"
            )

    # Length bias: short vs long sentences
    if "short_sents" in slice_stats and "long_sents" in slice_stats:
        short_overlap = slice_stats["short_sents"]["avg_word_overlap"]
        long_overlap  = slice_stats["long_sents"]["avg_word_overlap"]
        len_gap       = abs(short_overlap - long_overlap)
        print(f"Length word overlap gap (short vs long): {len_gap:.3f}")
        if len_gap > 0.05:
            bias_summary.append(
                f"Length bias detected: short_sents overlap={short_overlap:.3f} "
                f"vs long_sents={long_overlap:.3f} (gap={len_gap:.3f})"
            )

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "overall_stats":      overall_stats,
        "slice_stats":        slice_stats,
        "bias_scores":        bias_scores,
        "bias_threshold":     BIAS_THRESHOLD,
        "bias_flags":         bias_flags,
        "bias_summary":       bias_summary,
        "total_slices":       len(slice_stats),
        "biased_slices":      len(bias_flags),
        "bias_detected":      len(bias_flags) > 0,
        "cross_slice_analysis": {
            "domain_bias":     "medical" in slice_stats and "legal" in slice_stats,
            "direction_bias":  "en_to_es" in slice_stats and "es_to_en" in slice_stats,
            "length_bias":     "short_sents" in slice_stats and "long_sents" in slice_stats,
        }
    }

    report_path = os.path.join(REPORTS_DIR, "bias_report.json")
    json.dump(report, open(report_path, "w"), indent=2)
    print(f"\nBias report saved to {report_path}")

    # Print summary
    print("\n── Bias Detection Summary ──")
    if bias_summary:
        for item in bias_summary:
            print(f"  ⚠ {item}")
    else:
        print("  ✓ No significant bias detected across slices.")

    print(f"\nResult: {len(bias_flags)}/{len(slice_stats)} slices flagged for bias.")
    print("Node 6b complete.")


if __name__ == "__main__":
    detect_bias()
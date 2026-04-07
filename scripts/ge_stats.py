import os
import json

def generate_ge_stats():
    os.makedirs("/opt/airflow/reports", exist_ok=True)
    print("Reading train.jsonl line by line...")

    total = 0
    domain_counts = {}
    direction_counts = {}
    input_lens = []
    output_lens = []
    null_inputs = 0
    null_outputs = 0

    with open("/opt/airflow/data/processed/train.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total += 1

            domain = r.get("domain", "unknown")
            direction = r.get("direction", "unknown")
            inp = r.get("input", "")
            out = r.get("output", "")

            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            direction_counts[direction] = direction_counts.get(direction, 0) + 1

            if not inp:
                null_inputs += 1
            else:
                input_lens.append(len(inp.split()))
            if not out:
                null_outputs += 1
            else:
                output_lens.append(len(out.split()))

            if total % 100000 == 0:
                print(f"Processed {total} records...")

    summary = {
        "total_records": total,
        "domain_distribution": domain_counts,
        "direction_distribution": direction_counts,
        "input_len_mean": round(sum(input_lens) / len(input_lens), 2) if input_lens else 0,
        "input_len_max": max(input_lens) if input_lens else 0,
        "input_len_min": min(input_lens) if input_lens else 0,
        "output_len_mean": round(sum(output_lens) / len(output_lens), 2) if output_lens else 0,
        "output_len_max": max(output_lens) if output_lens else 0,
        "null_input_count": null_inputs,
        "null_output_count": null_outputs,
    }

    schema = {
        "required_fields": ["instruction", "input", "output", "domain", "direction"],
        "valid_domains": ["medical", "legal"],
        "valid_directions": ["en_to_es", "es_to_en"],
        "total_records": total,
        "null_counts": {"input": null_inputs, "output": null_outputs},
        "input_len_stats": {
            "mean": summary["input_len_mean"],
            "max": summary["input_len_max"],
            "min": summary["input_len_min"],
        }
    }

    json.dump(summary, open("/opt/airflow/reports/tfdv_stats_summary.json", "w"), indent=2)
    json.dump(schema, open("/opt/airflow/reports/schema.json", "w"), indent=2)

    print("Stats written to reports/tfdv_stats_summary.json")
    print(json.dumps(summary, indent=2))
    print("Node 4 complete.")
import os
import json

MAX_INPUT_LEN = 200
MIN_INPUT_LEN = 3
VALID_DOMAINS = ["medical", "legal"]
VALID_DIRECTIONS = ["en_to_es", "es_to_en"]

def detect_anomalies():
    os.makedirs("/opt/airflow/reports", exist_ok=True)
    print("Reading val.jsonl for anomaly detection...")

    anomalies_found = []
    total = 0
    too_long = 0
    too_short = 0
    invalid_domains = 0
    invalid_dirs = 0
    null_inputs = 0
    null_outputs = 0

    with open("/opt/airflow/data/processed/val.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total += 1

            inp = r.get("input", "")
            out = r.get("output", "")
            domain = r.get("domain", "")
            direction = r.get("direction", "")

            if not inp:
                null_inputs += 1
            else:
                inp_len = len(inp.split())
                if inp_len > MAX_INPUT_LEN:
                    too_long += 1
                if inp_len < MIN_INPUT_LEN:
                    too_short += 1

            if not out:
                null_outputs += 1
            if domain not in VALID_DOMAINS:
                invalid_domains += 1
            if direction not in VALID_DIRECTIONS:
                invalid_dirs += 1

    if too_long > 0:
        anomalies_found.append(f"Sentences too long: {too_long} records")
    if too_short > 0:
        anomalies_found.append(f"Sentences too short: {too_short} records")
    if invalid_domains > 0:
        anomalies_found.append(f"Invalid domains: {invalid_domains} records")
    if invalid_dirs > 0:
        anomalies_found.append(f"Invalid directions: {invalid_dirs} records")
    if null_inputs > 0:
        anomalies_found.append(f"Null inputs: {null_inputs} records")
    if null_outputs > 0:
        anomalies_found.append(f"Null outputs: {null_outputs} records")

    json.dump(
        {"anomalies": anomalies_found, "total_checked": total},
        open("/opt/airflow/reports/anomalies.json", "w"),
        indent=2,
    )

    if anomalies_found:
        print(f"ANOMALIES DETECTED: {anomalies_found}")
        raise ValueError(f"Dataset anomalies detected: {anomalies_found}")

    print(f"No anomalies detected. Validated {total} records.")
    print("Node 6 complete.")
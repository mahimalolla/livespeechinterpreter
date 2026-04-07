import os
import json
 
MAX_INPUT_LEN = 200
MIN_INPUT_LEN = 3
VALID_DOMAINS = ["medical", "legal"]
VALID_DIRECTIONS = ["en_to_es", "es_to_en"]
 
def detect_anomalies():
    # If anomalies are excessive, drop the invalid records and continue.
    # This is a "repair gate" rather than a hard fail for large-scale issues.
    drop_threshold = int(os.environ.get("ANOMALY_DROP_THRESHOLD", "1000"))
 
    os.makedirs("/opt/airflow/reports", exist_ok=True)
    os.makedirs("/opt/airflow/data/processed", exist_ok=True)
    print("Reading val.jsonl for anomaly detection...")
 
    anomalies_found = []
    total = 0
 
    # Category counts (a single record can contribute to multiple categories)
    too_long = 0
    too_short = 0
    invalid_domains = 0
    invalid_dirs = 0
    null_inputs = 0
    null_outputs = 0
 
    # Row-level anomalous count (union of category checks)
    anomalous_rows = 0
    cleaned_records = []
 
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
 
            row_is_anomalous = False
 
            if not inp:
                null_inputs += 1
                row_is_anomalous = True
            else:
                inp_len = len(inp.split())
                if inp_len > MAX_INPUT_LEN:
                    too_long += 1
                    row_is_anomalous = True
                if inp_len < MIN_INPUT_LEN:
                    too_short += 1
                    row_is_anomalous = True
 
            if not out:
                null_outputs += 1
                row_is_anomalous = True
 
            if domain not in VALID_DOMAINS:
                invalid_domains += 1
                row_is_anomalous = True
 
            if direction not in VALID_DIRECTIONS:
                invalid_dirs += 1
                row_is_anomalous = True
 
            if row_is_anomalous:
                anomalous_rows += 1
            else:
                cleaned_records.append(r)
 
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
 
    report = {
        "anomalies": anomalies_found,
        "total_checked": total,
        "anomalous_row_count": anomalous_rows,
        "drop_threshold": drop_threshold,
        "action_taken": "none",
    }
 
    # Repair/drop strategy
    if anomalies_found and anomalous_rows > drop_threshold:
        out_path = "/opt/airflow/data/processed/val.jsonl"
        with open(out_path, "w") as out_f:
            for rec in cleaned_records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
 
        report["action_taken"] = "drop_and_continue"
        report["records_dropped"] = anomalous_rows
        report["records_after_drop"] = len(cleaned_records)
 
        print(f"ANOMALY THRESHOLD EXCEEDED ({anomalous_rows} > {drop_threshold}). Dropping invalid val rows.")
        print(f"Overwrote {out_path} with {len(cleaned_records)} valid records.")
    elif anomalies_found:
        report["action_taken"] = "fail_fast"
        report["records_dropped"] = 0
        report["records_after_drop"] = total - anomalous_rows
        print(f"ANOMALIES DETECTED: {anomalies_found}")
        json.dump(report, open("/opt/airflow/reports/anomalies.json", "w"), indent=2)
        raise ValueError(f"Dataset anomalies detected: {anomalies_found}")
    else:
        report["action_taken"] = "validated"
 
    json.dump(report, open("/opt/airflow/reports/anomalies.json", "w"), indent=2)
 
    if not anomalies_found:
        print(f"No anomalies detected. Validated {total} records.")
    else:
        print(f"Anomalies found: {anomalous_rows} rows. Action: {report['action_taken']}")
    print("Node 6 complete.")

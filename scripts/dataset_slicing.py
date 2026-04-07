import os
import json

def create_slices():
    os.makedirs("/opt/airflow/data/processed/slices", exist_ok=True)
    print("Reading test.jsonl for slicing...")

    slices = {
        "medical": [],
        "legal": [],
        "en_to_es": [],
        "es_to_en": [],
        "short_sents": [],
        "long_sents": [],
    }

    with open("/opt/airflow/data/processed/test.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            inp_len = len(r.get("input", "").split())

            if r.get("domain") == "medical":
                slices["medical"].append(r)
            if r.get("domain") == "legal":
                slices["legal"].append(r)
            if r.get("direction") == "en_to_es":
                slices["en_to_es"].append(r)
            if r.get("direction") == "es_to_en":
                slices["es_to_en"].append(r)
            if inp_len <= 15:
                slices["short_sents"].append(r)
            if inp_len > 30:
                slices["long_sents"].append(r)

    for name, data in slices.items():
        path = f"/opt/airflow/data/processed/slices/{name}.jsonl"
        with open(path, "w") as out:
            for r in data:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Slice [{name}]: {len(data)} records → {path}")

    print("Node 7 complete.")
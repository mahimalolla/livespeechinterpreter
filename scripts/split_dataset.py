import os
import json
import random

def split_dataset():
    print("Loading dataset.jsonl...")
    records = []
    with open("/opt/airflow/data/processed/dataset.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    random.seed(42)
    random.shuffle(records)

    n = len(records)
    train_end = int(0.80 * n)
    val_end = int(0.90 * n)

    splits = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }

    for split, data in splits.items():
        path = f"/opt/airflow/data/processed/{split}.jsonl"
        with open(path, "w") as out:
            for r in data:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(data)} records → {path}")

    print(f"Total: {n} | Train: {train_end} | Val: {val_end-train_end} | Test: {n-val_end}")
    print("Node 3 complete.")
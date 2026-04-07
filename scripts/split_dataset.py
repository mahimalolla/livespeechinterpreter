import os
import json
import random
 
def split_dataset():
    """
    Create train/val/test splits while preserving slice composition.
 
    This reduces "slice imbalance" issues flagged later by bias detection by keeping
    domain + direction + sentence-length buckets consistent across splits.
    """
    print("Loading dataset.jsonl...")
    records = []
    with open("/opt/airflow/data/processed/dataset.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
 
    def length_bucket(inp_text: str) -> str:
        inp_len = len((inp_text or "").split())
        # Match the slicing thresholds used later in dataset_slicing.py
        if inp_len <= 15:
            return "short_sents"
        if inp_len > 30:
            return "long_sents"
        return "mid_sents"
 
    # Ratios used by the pipeline
    train_ratio = 0.80
    val_ratio = 0.10
    test_ratio = 0.10
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.")
 
    random.seed(42)
 
    # Group by slice-defining characteristics so each split gets similar composition
    groups = {}
    for r in records:
        key = (
            r.get("domain", "unknown"),
            r.get("direction", "unknown"),
            length_bucket(r.get("input", "")),
        )
        groups.setdefault(key, []).append(r)
 
    splits = {"train": [], "val": [], "test": []}
 
    for _, group_records in groups.items():
        group_records = list(group_records)
        random.shuffle(group_records)
        g = len(group_records)
        if g == 0:
            continue
 
        train_end = int(train_ratio * g)
        val_end = train_end + int(val_ratio * g)
 
        # Ensure non-empty allocations when possible
        if g >= 3:
            train_end = max(train_end, 1)
            val_end = max(val_end, train_end + 1)
 
        splits["train"].extend(group_records[:train_end])
        splits["val"].extend(group_records[train_end:val_end])
        splits["test"].extend(group_records[val_end:])
 
    # Shuffle each split so examples aren't clustered by slice
    for split_name, data in splits.items():
        random.shuffle(data)
        path = f"/opt/airflow/data/processed/{split_name}.jsonl"
        with open(path, "w") as out:
            for r in data:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split_name}: {len(data)} records → {path}")
 
    n = len(records)
    print(f"Total: {n} | Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
    print("Node 3 complete.")

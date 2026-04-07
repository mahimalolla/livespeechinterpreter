import os
import json
import hashlib
from datasets import load_dataset
from tqdm import tqdm

DATASETS = {
    "emea_medical": {
        "hf_id": "Helsinki-NLP/opus-100",
        "config": "en-es",
        "domain": "medical",
    },
    "europarl_legal": {
        "hf_id": "Helsinki-NLP/europarl",
        "config": "en-es",
        "domain": "legal",
    },
}

def download_datasets():
    os.makedirs("/opt/airflow/data/raw", exist_ok=True)
    manifest = {}

    for name, cfg in DATASETS.items():
        print(f"Downloading {name}...")
        ds = load_dataset(cfg["hf_id"], cfg["config"], trust_remote_code=True)
        out_path = f"/opt/airflow/data/raw/{name}.jsonl"
        ds["train"].to_json(out_path)
        checksum = hashlib.md5(open(out_path, "rb").read()).hexdigest()
        manifest[name] = {
            "path": out_path,
            "checksum": checksum,
            "domain": cfg["domain"],
        }
        print(f"Saved {name} → {out_path} | checksum: {checksum}")

    json.dump(manifest, open("/opt/airflow/data/raw/manifest.json", "w"), indent=2)
    print("Manifest saved.")
    print("Node 1 complete.")
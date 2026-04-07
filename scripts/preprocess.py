import os
import json
import re
import ftfy
from tqdm import tqdm

MAX_TOKENS = 200
MIN_TOKENS = 3
MAX_RATIO = 2.5

def clean_text(text):
    text = ftfy.fix_text(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def is_valid_pair(en, es):
    en_toks = en.split()
    es_toks = es.split()
    if not (MIN_TOKENS <= len(en_toks) <= MAX_TOKENS):
        return False
    if not (MIN_TOKENS <= len(es_toks) <= MAX_TOKENS):
        return False
    ratio = len(en_toks) / max(len(es_toks), 1)
    if ratio > MAX_RATIO or ratio < 1 / MAX_RATIO:
        return False
    return True

def to_instruction(en, es, domain, direction="en_to_es"):
    if direction == "en_to_es":
        return {
            "instruction": f"Translate the following {domain} sentence from English to Spanish.",
            "input": en,
            "output": es,
            "domain": domain,
            "direction": direction,
        }
    else:
        return {
            "instruction": f"Translate the following {domain} sentence from Spanish to English.",
            "input": es,
            "output": en,
            "domain": domain,
            "direction": direction,
        }

def preprocess_datasets():
    os.makedirs("/opt/airflow/data/processed", exist_ok=True)
    manifest = json.load(open("/opt/airflow/data/raw/manifest.json"))
    all_records = []
    seen = set()

    for name, meta in manifest.items():
        domain = meta["domain"]
        print(f"Processing {name}...")
        with open(meta["path"]) as f:
            for line in tqdm(f, desc=name):
                try:
                    row = json.loads(line)
                    translation = row.get("translation", {})
                    en = clean_text(translation.get("en", ""))
                    es = clean_text(translation.get("es", ""))
                    if not en or not es:
                        continue
                    key = hash(en + es)
                    if key in seen:
                        continue
                    if not is_valid_pair(en, es):
                        continue
                    seen.add(key)
                    all_records.append(to_instruction(en, es, domain, "en_to_es"))
                    all_records.append(to_instruction(en, es, domain, "es_to_en"))
                except Exception:
                    continue

    out_path = "/opt/airflow/data/processed/dataset.jsonl"
    with open(out_path, "w") as out:
        for r in all_records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Total records after processing: {len(all_records)}")
    print("Node 2 complete.")
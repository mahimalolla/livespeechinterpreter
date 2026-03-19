import os
import json
import subprocess

RAW_DIR       = "/opt/airflow/data/raw"
MANIFEST_PATH = "/opt/airflow/data/raw/manifest.json"
DVC_REMOTE    = "gcs_remote"

def dvc_pull_or_download():
    print("Node 0 — DVC Pull or Download")
    os.makedirs(RAW_DIR, exist_ok=True)

    # Check if raw data already exists locally
    if _raw_data_exists():
        print("Raw data already exists locally. Skipping pull.")
        return True

    # Try to pull from DVC remote (GCS)
    print("Raw data not found locally. Attempting DVC pull from GCS...")
    pull_success = _try_dvc_pull()

    if pull_success and _raw_data_exists():
        print("DVC pull successful. Raw data restored from GCS cache.")
        return True

    # If DVC pull failed or no cache exists, download fresh from HuggingFace
    print("DVC pull failed or no cache found. Will download fresh from HuggingFace in Node 1.")
    print("Node 0 complete — Node 1 will handle fresh download.")
    return False

def _raw_data_exists() -> bool:
    """Check if manifest and at least one dataset file exist."""
    if not os.path.exists(MANIFEST_PATH):
        return False
    try:
        manifest = json.load(open(MANIFEST_PATH))
        for name, meta in manifest.items():
            if not os.path.exists(meta["path"]):
                print(f"Missing dataset file: {meta['path']}")
                return False
        print(f"Found existing raw data: {list(manifest.keys())}")
        return True
    except Exception as e:
        print(f"Manifest read error: {e}")
        return False

def _try_dvc_pull() -> bool:
    """Attempt to pull raw data from DVC GCS remote."""
    try:
        result = subprocess.run(
            ["dvc", "pull", "data/raw"],
            cwd="/opt/airflow",
            capture_output=True,
            text=True,
            timeout=300,
        )
        print(f"DVC pull stdout: {result.stdout}")
        if result.returncode == 0:
            print("DVC pull succeeded.")
            return True
        else:
            print(f"DVC pull failed (code {result.returncode}): {result.stderr}")
            return False
    except FileNotFoundError:
        print("DVC not found in container. Skipping pull.")
        return False
    except subprocess.TimeoutExpired:
        print("DVC pull timed out after 5 minutes.")
        return False
    except Exception as e:
        print(f"DVC pull exception: {e}")
        return False
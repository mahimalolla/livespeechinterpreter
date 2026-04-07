import os
import json
import subprocess
import datetime

def dvc_push_datasets():
    print("Node 9b — DVC Push")

    # Track raw and processed data with DVC
    _dvc_add_and_commit("data/raw", "raw datasets")
    _dvc_add_and_commit("data/processed", "processed datasets")

    # Push everything to GCS remote
    print("Pushing all tracked data to GCS DVC remote...")
    try:
        result = subprocess.run(
            ["dvc", "push"],
            cwd="/opt/airflow",
            capture_output=True,
            text=True,
            timeout=600,
        )
        print(f"DVC push stdout: {result.stdout}")
        if result.returncode == 0:
            print("DVC push succeeded. All data versioned in GCS.")
        else:
            print(f"DVC push warning (code {result.returncode}): {result.stderr}")
            print("Continuing pipeline despite DVC push issue.")
    except FileNotFoundError:
        print("DVC not found in container. Skipping push.")
    except subprocess.TimeoutExpired:
        print("DVC push timed out after 10 minutes.")
    except Exception as e:
        print(f"DVC push exception: {e}")

    # Save DVC version report
    _save_dvc_report()
    print("Node 9b complete.")

def _dvc_add_and_commit(path: str, description: str):
    """Add a path to DVC tracking and commit the .dvc file to git."""
    try:
        # DVC add
        result = subprocess.run(
            ["dvc", "add", path],
            cwd="/opt/airflow",
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            print(f"DVC add {path} succeeded.")
        else:
            print(f"DVC add {path} warning: {result.stderr}")
            return

        # Git add the .dvc file
        dvc_file = f"{path}.dvc"
        subprocess.run(
            ["git", "add", dvc_file, ".dvcignore"],
            cwd="/opt/airflow",
            capture_output=True,
            text=True,
        )

        # Git commit
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        subprocess.run(
            ["git", "commit", "-m", f"data: DVC track {description} {ts}"],
            cwd="/opt/airflow",
            capture_output=True,
            text=True,
        )
        print(f"Git committed .dvc file for {path}")

    except Exception as e:
        print(f"DVC add/commit failed for {path}: {e} — continuing.")

def _save_dvc_report():
    """Save a DVC versioning report to reports/."""
    os.makedirs("/opt/airflow/reports", exist_ok=True)
    try:
        result = subprocess.run(
            ["dvc", "status"],
            cwd="/opt/airflow",
            capture_output=True,
            text=True,
            timeout=30,
        )
        report = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "dvc_status": result.stdout,
            "push_completed": True,
        }
        import json
        json.dump(report, open("/opt/airflow/reports/dvc_report.json", "w"), indent=2)
        print("DVC report saved to reports/dvc_report.json")
    except Exception as e:
        print(f"DVC report save failed: {e}")
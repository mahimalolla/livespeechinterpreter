#!/usr/bin/env python3
"""
Register an already-trained model (e.g. Vertex / GCS adapter) with MLflow
without retraining: creates a run under experiment gemma3-translation and
optionally registers a Model Registry entry whose artifact points at gs://...

Requires: pip install mlflow pandas
Auth: GOOGLE_APPLICATION_CREDENTIALS (or gcloud ADC) so MLflow can upload
artifacts to the server's default artifact root (GCS).

Note: Cloud MLflow is 2.12.x. Newer MLflow clients use /logged-models when calling
log_model(), which those servers do not implement (404). This script uses
save_model + log_artifacts instead. Optional: pip install 'mlflow==2.12.1' to match the server.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd

DEFAULT_TRACKING = "https://mlflow-server-1050963407386.us-central1.run.app"
DEFAULT_GCS = "gs://livespeechinterpreter-training/models/gemma3-4b-translation-v17"


class GcsPointerModel(mlflow.pyfunc.PythonModel):
    """Minimal pyfunc so MLflow has a valid model/ dir; real weights stay on GCS."""

    def load_context(self, context):
        path = context.artifacts["gcs_pointer"]
        self.gcs_uri = Path(path).read_text().strip()

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "model_gcs": [self.gcs_uri],
                "note": ["Load adapter from this URI in inference (HF / your stack)."],
            }
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill MLflow run + registry for existing GCS model")
    p.add_argument(
        "--gcs-uri",
        default=os.environ.get("MODEL_GCS_URI", DEFAULT_GCS),
        help="GCS prefix where the trained adapter / model lives",
    )
    p.add_argument(
        "--experiment",
        default="gemma3-translation",
        help="Must match training/train.py",
    )
    p.add_argument(
        "--run-name",
        default="",
        help="Defaults to vertex-import-<timestamp> if empty",
    )
    p.add_argument(
        "--registered-model-name",
        default="gemma3-4b-translation",
        help="Name in MLflow Models (Model Registry); create new version if exists",
    )
    p.add_argument(
        "--vertex-job",
        default="",
        help="Optional Vertex job or pipeline resource name for traceability",
    )
    p.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING),
    )
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    from datetime import datetime, timezone

    run_name = args.run_name or f"vertex-import-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pointer = root / "gcs_uri.txt"
        pointer.write_text(args.gcs_uri)
        staging = root / "model_staging"
        staging.mkdir()
        mlflow.pyfunc.save_model(
            path=str(staging),
            python_model=GcsPointerModel(),
            artifacts={"gcs_pointer": str(pointer)},
        )

        with mlflow.start_run(run_name=run_name) as run:
            gcs_parts = args.gcs_uri.replace("gs://", "").split("/", 1)
            gcs_bucket = gcs_parts[0] if gcs_parts else ""
            gcs_object_prefix = gcs_parts[1] if len(gcs_parts) > 1 else ""

            mlflow.log_params(
                {
                    "analysis_type": "registry_import",
                    "source": "vertex_ai_existing",
                    "adapter_gcs": args.gcs_uri,
                    "gcs_bucket": gcs_bucket,
                    "gcs_object_prefix": gcs_object_prefix,
                    "vertex_job": args.vertex_job or "not_set",
                    "registered_as": args.registered_model_name,
                }
            )

            try:
                repo_root = Path(__file__).resolve().parents[1]
                sha = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(repo_root),
                    text=True,
                    timeout=5,
                ).strip()
                mlflow.set_tag("mlflow.source.git.commit", sha[:12])
            except (subprocess.CalledProcessError, FileNotFoundError, OSError):
                pass

            manifest = {
                "adapter_gcs": args.gcs_uri,
                "registered_model_name": args.registered_model_name,
                "vertex_job": args.vertex_job or None,
            }
            manifest_path = root / "registry_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))
            mlflow.log_artifact(str(manifest_path))

            mlflow.log_artifacts(str(staging), artifact_path="model")
            model_uri = f"runs:/{run.info.run_id}/model"
            mv = mlflow.register_model(model_uri=model_uri, name=args.registered_model_name)

            elapsed = time.perf_counter() - t0
            bundle_files = sum(1 for f in staging.rglob("*") if f.is_file())
            mlflow.log_metrics(
                {
                    "import_completed": 1.0,
                    "registry_version": float(mv.version),
                    "elapsed_seconds": elapsed,
                    "model_bundle_files": float(bundle_files),
                }
            )

            print(f"Run: {run.info.run_id}")
            print(f"Registered model: {args.registered_model_name} version {mv.version}")
            print(f"Open Experiments → {args.experiment} and Models → {args.registered_model_name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Completed Vertex AI training run into MLflow (no retraining).

Use this when training happened on Vertex and you want one MLflow run that
captures: job IDs, hyperparameters, and final training metrics (e.g. loss).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow

DEFAULT_TRACKING = "https://mlflow-server-1050963407386.us-central1.run.app"
DEFAULT_EXPERIMENT = "gemma3-translation"


def main() -> None:
    p = argparse.ArgumentParser(description="Vertex training metadata to MLflow")
    p.add_argument("--tracking-uri", default=DEFAULT_TRACKING)
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    p.add_argument("--run-name", default="")

    # Vertex provenance
    p.add_argument("--vertex-pipeline-id", default="")
    p.add_argument("--vertex-custom-job-id", default="")
    p.add_argument("--vertex-region", default="us-central1")

    # Model/data refs
    p.add_argument("--base-model", default="google/gemma-3-4b-it")
    p.add_argument("--adapter-gcs", required=True)
    p.add_argument("--train-gcs", default="")
    p.add_argument("--val-gcs", default="")

    # Hyperparameters (as used in training/train.py)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=float, default=4.0)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--lora-r", type=float, default=16.0)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-seq-length", type=float, default=128.0)
    p.add_argument("--grad-accum", type=float, default=4.0)

    # Metrics from Vertex logs (you can pass only what you have)
    p.add_argument("--train-loss-final", type=float, required=True)
    p.add_argument("--train-loss-start", type=float, default=-1.0)
    p.add_argument("--train-loss-best", type=float, default=-1.0)
    p.add_argument("--train-steps", type=float, default=-1.0)
    p.add_argument("--train-hours", type=float, default=-1.0)
    p.add_argument("--notes", default="")
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    run_name = args.run_name or f"vertex-train-backfill-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "source": "vertex_ai_backfill",
                "training_domains": "medical,legal",
                "task": "en_es_instruction_translation",
            }
        )

        params = {
            "base_model": args.base_model,
            "adapter_gcs": args.adapter_gcs,
            "train_gcs": args.train_gcs or "not_set",
            "val_gcs": args.val_gcs or "not_set",
            "vertex_pipeline_id": args.vertex_pipeline_id or "not_set",
            "vertex_custom_job_id": args.vertex_custom_job_id or "not_set",
            "vertex_region": args.vertex_region,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "max_seq_length": args.max_seq_length,
            "grad_accum": args.grad_accum,
        }
        mlflow.log_params(params)

        metrics = {"train_loss_final": args.train_loss_final}
        if args.train_loss_start >= 0:
            metrics["train_loss_start"] = args.train_loss_start
            metrics["train_loss_delta"] = args.train_loss_start - args.train_loss_final
        if args.train_loss_best >= 0:
            metrics["train_loss_best"] = args.train_loss_best
        if args.train_steps >= 0:
            metrics["train_steps"] = args.train_steps
        if args.train_hours >= 0:
            metrics["train_hours"] = args.train_hours
        mlflow.log_metrics(metrics)

        artifact = {
            "summary": "Vertex AI pipeline/logs",
            "notes": args.notes,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "params": params,
            "metrics": metrics,
        }
        out = Path("/tmp/vertex_training_summary.json")
        out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(out))

    print(f"Logged MLflow run '{run_name}' in experiment '{args.experiment}'.")


if __name__ == "__main__":
    main()

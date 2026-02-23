"""
TFDV validation for the NMT (Neural Machine Translation) pipeline.

Validates both raw and processed NMT data by generating statistics,
inferring schemas, and detecting anomalies such as missing features,
distribution drift, and unexpected values.
"""

import os
import sys
import json
import logging
import pandas as pd
import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2, anomalies_pb2
from google.protobuf import text_format

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "..", ".."))


def flatten_nmt_for_validation(df):
    """Flatten token-list columns into scalar features TFDV can profile."""
    flat = pd.DataFrame()
    flat["en"] = df["en"].astype(str) if "en" in df.columns else pd.Series(dtype=str)
    flat["es"] = df["es"].astype(str) if "es" in df.columns else pd.Series(dtype=str)
    flat["en_word_count"] = flat["en"].str.split().str.len()
    flat["es_word_count"] = flat["es"].str.split().str.len()
    flat["en_char_count"] = flat["en"].str.len()
    flat["es_char_count"] = flat["es"].str.len()

    if "en_tokens" in df.columns:
        flat["en_token_count"] = df["en_tokens"].apply(
            lambda x: len(x) if isinstance(x, (list, tuple)) else 0
        )
    if "es_tokens" in df.columns:
        flat["es_token_count"] = df["es_tokens"].apply(
            lambda x: len(x) if isinstance(x, (list, tuple)) else 0
        )

    return flat


def save_anomalies_report(anomalies, path):
    """Persist anomaly proto as human-readable text and JSON summary."""
    text_path = path + ".txt"
    json_path = path + ".json"

    proto_text = text_format.MessageToString(anomalies)
    with open(text_path, "w") as f:
        f.write(proto_text)
    logging.info(f"Anomaly report (text) saved to {text_path}")

    summary = {
        "anomaly_count": len(anomalies.anomaly_info),
        "features_with_anomalies": [],
    }
    for feature_name, info in anomalies.anomaly_info.items():
        summary["features_with_anomalies"].append({
            "feature": feature_name,
            "short_description": info.short_description,
            "description": info.description,
            "severity": anomalies_pb2.AnomalyInfo.Severity.Name(info.severity),
        })
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Anomaly report (JSON) saved to {json_path}")

    return summary


def validate_raw_nmt(project_root, output_dir):
    """Generate stats and schema for raw NMT data (OPUS + domain)."""
    logging.info("=" * 60)
    logging.info("TFDV: Validating RAW NMT data")
    logging.info("=" * 60)

    opus_path = os.path.join(project_root, "data", "raw", "opus", "opus_100k.parquet")
    domain_path = os.path.join(project_root, "data", "raw", "domain_data", "domain_specific_raw.parquet")

    dfs = []
    for label, path in [("opus", opus_path), ("domain", domain_path)]:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            logging.info(f"Loaded {label}: {len(df)} rows, columns={list(df.columns)}")
            dfs.append(df)
        else:
            logging.warning(f"Raw file not found: {path}")

    if not dfs:
        logging.error("No raw NMT data found. Skipping raw validation.")
        return None, None

    combined = pd.concat(dfs, ignore_index=True)
    flat = flatten_nmt_for_validation(combined)

    logging.info(f"Generating statistics for {len(flat)} raw NMT rows...")
    stats = tfdv.generate_statistics_from_dataframe(flat)

    stats_path = os.path.join(output_dir, "raw_nmt_stats.pb")
    tfdv.write_stats_text(stats, stats_path)
    logging.info(f"Raw NMT statistics saved to {stats_path}")

    schema = tfdv.infer_schema(stats)

    tfdv.set_domain(schema, "en_word_count", schema_pb2.IntDomain(min=1, max=500))
    tfdv.set_domain(schema, "es_word_count", schema_pb2.IntDomain(min=1, max=500))

    schema_path = os.path.join(output_dir, "raw_nmt_schema.pbtxt")
    tfdv.write_schema_text(schema, schema_path)
    logging.info(f"Raw NMT schema saved to {schema_path}")

    return stats, schema


def validate_processed_nmt(project_root, output_dir, raw_schema=None):
    """
    Validate processed NMT data against the raw schema to detect drift
    or anomalies introduced during preprocessing.
    """
    logging.info("=" * 60)
    logging.info("TFDV: Validating PROCESSED NMT data")
    logging.info("=" * 60)

    processed_path = os.path.join(project_root, "data", "processed", "nmt_processed.parquet")
    if not os.path.exists(processed_path):
        logging.error(f"Processed NMT file not found: {processed_path}")
        return None

    df = pd.read_parquet(processed_path)
    logging.info(f"Loaded processed NMT: {len(df)} rows, columns={list(df.columns)}")

    flat = flatten_nmt_for_validation(df)

    logging.info(f"Generating statistics for {len(flat)} processed NMT rows...")
    stats = tfdv.generate_statistics_from_dataframe(flat)

    stats_path = os.path.join(output_dir, "processed_nmt_stats.pb")
    tfdv.write_stats_text(stats, stats_path)
    logging.info(f"Processed NMT statistics saved to {stats_path}")

    processed_schema = tfdv.infer_schema(stats)
    schema_path = os.path.join(output_dir, "processed_nmt_schema.pbtxt")
    tfdv.write_schema_text(processed_schema, schema_path)
    logging.info(f"Processed NMT schema saved to {schema_path}")

    if raw_schema is not None:
        logging.info("Checking for anomalies against raw NMT schema...")
        anomalies = tfdv.validate_statistics(stats, raw_schema)
        summary = save_anomalies_report(
            anomalies, os.path.join(output_dir, "nmt_anomalies")
        )
        if summary["anomaly_count"] > 0:
            logging.warning(
                f"TFDV detected {summary['anomaly_count']} anomalies in processed NMT data!"
            )
            for a in summary["features_with_anomalies"]:
                logging.warning(f"  - {a['feature']}: {a['short_description']}")
        else:
            logging.info("No anomalies detected in processed NMT data.")
    else:
        logging.info("No raw schema available; skipping cross-stage anomaly detection.")
        anomalies = tfdv.validate_statistics(stats, processed_schema)
        save_anomalies_report(anomalies, os.path.join(output_dir, "nmt_anomalies"))

    return stats


def main():
    project_root = get_project_root()
    output_dir = os.path.join(project_root, "data", "validation", "nmt")
    os.makedirs(output_dir, exist_ok=True)

    raw_stats, raw_schema = validate_raw_nmt(project_root, output_dir)
    validate_processed_nmt(project_root, output_dir, raw_schema=raw_schema)

    logging.info("NMT TFDV validation complete.")


if __name__ == "__main__":
    main()

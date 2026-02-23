"""
TFDV validation for the ASR (Automatic Speech Recognition) pipeline.

Validates both raw and processed ASR data. Since TFDV cannot profile
nested arrays (e.g. mel spectrograms), this script extracts scalar
metadata features for TFDV and logs spectrogram-level statistics
separately.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2, anomalies_pb2
from google.protobuf import text_format

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "..", ".."))


def flatten_raw_asr(df):
    """Extract scalar features from raw ASR data for TFDV."""
    flat = pd.DataFrame()

    if "text" in df.columns:
        flat["text"] = df["text"].astype(str)
        flat["text_word_count"] = flat["text"].str.split().str.len()
        flat["text_char_count"] = flat["text"].str.len()

    if "sampling_rate" in df.columns:
        flat["sampling_rate"] = df["sampling_rate"].astype(float)

    if "audio_array" in df.columns:
        flat["audio_length"] = df["audio_array"].apply(
            lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
        )
        flat["audio_duration_sec"] = flat.get("audio_length", 0) / flat.get("sampling_rate", 16000)

    if "chapter_id" in df.columns:
        flat["chapter_id"] = df["chapter_id"].astype(str)
    if "speaker_id" in df.columns:
        flat["speaker_id"] = df["speaker_id"].astype(str)

    return flat


def flatten_processed_asr(df):
    """Extract scalar features from processed ASR data for TFDV."""
    flat = pd.DataFrame()

    if "text" in df.columns:
        flat["text"] = df["text"].astype(str)
        flat["text_word_count"] = flat["text"].str.split().str.len()
        flat["text_char_count"] = flat["text"].str.len()

    if "sampling_rate" in df.columns:
        flat["sampling_rate"] = df["sampling_rate"].astype(float)

    if "log_mel_spec" in df.columns:
        flat["mel_num_bins"] = df["log_mel_spec"].apply(
            lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
        )
        flat["mel_num_frames"] = df["log_mel_spec"].apply(
            lambda x: len(x[0]) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0
        )

    if "chapter_id" in df.columns:
        flat["chapter_id"] = df["chapter_id"].astype(str)
    if "speaker_id" in df.columns:
        flat["speaker_id"] = df["speaker_id"].astype(str)

    return flat


def compute_spectrogram_stats(processed_path, output_dir):
    """Compute custom statistics for mel spectrograms that TFDV can't handle."""
    import pyarrow.parquet as pq

    if not os.path.exists(processed_path):
        logging.info("Processed ASR file not found; skipping spectrogram stats.")
        return

    pf = pq.ParquetFile(processed_path)
    schema_str = str(pf.schema)
    if "log_mel_spec" not in schema_str:
        logging.info("No log_mel_spec column found; skipping spectrogram stats.")
        return

    logging.info("Computing spectrogram-level statistics (outside TFDV)...")
    logging.info("Loading 20 spectrograms for profiling...")
    try:
        batch = next(pf.iter_batches(batch_size=20))
        spec_df = batch.to_pandas()
        if "log_mel_spec" not in spec_df.columns:
            logging.warning("log_mel_spec column not found in DataFrame; skipping.")
            return
        sample = spec_df["log_mel_spec"]
        sample_size = len(sample)
    except Exception as e:
        logging.warning(f"Could not load spectrograms for profiling: {e}")
        return

    all_means = []
    all_stds = []
    all_mins = []
    all_maxs = []
    frame_counts = []

    for spec in sample:
        if isinstance(spec, (list, np.ndarray)):
            try:
                if isinstance(spec, np.ndarray) and spec.dtype == object:
                    arr_2d = np.vstack(spec).astype(np.float32)
                elif isinstance(spec, np.ndarray) and spec.ndim == 2:
                    arr_2d = spec.astype(np.float32)
                elif isinstance(spec, list):
                    arr_2d = np.array([np.array(row, dtype=np.float32) for row in spec])
                else:
                    continue

                all_means.append(float(np.mean(arr_2d)))
                all_stds.append(float(np.std(arr_2d)))
                all_mins.append(float(np.min(arr_2d)))
                all_maxs.append(float(np.max(arr_2d)))
                frame_counts.append(arr_2d.shape[1] if arr_2d.ndim == 2 else 0)
            except Exception as e:
                logging.warning(f"Skipping one spectrogram sample: {e}")

    spec_stats = {
        "sample_size": sample_size,
        "mean": {"avg": np.mean(all_means), "std": np.std(all_means)},
        "std_dev": {"avg": np.mean(all_stds), "std": np.std(all_stds)},
        "min_value": {"avg": np.mean(all_mins), "min": np.min(all_mins)},
        "max_value": {"avg": np.mean(all_maxs), "max": np.max(all_maxs)},
        "frame_count": {
            "avg": np.mean(frame_counts),
            "min": int(np.min(frame_counts)),
            "max": int(np.max(frame_counts)),
        },
    }

    def convert_numpy(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    stats_path = os.path.join(output_dir, "spectrogram_stats.json")
    with open(stats_path, "w") as f:
        json.dump(spec_stats, f, indent=2, default=convert_numpy)
    logging.info(f"Spectrogram statistics saved to {stats_path}")

    logging.info(f"  Mean dB (avg): {spec_stats['mean']['avg']:.2f}")
    logging.info(f"  Std dB (avg):  {spec_stats['std_dev']['avg']:.2f}")
    logging.info(f"  Frames (avg):  {spec_stats['frame_count']['avg']:.0f}")
    logging.info(f"  Frames range:  [{spec_stats['frame_count']['min']}, {spec_stats['frame_count']['max']}]")


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


def validate_raw_asr(project_root, output_dir):
    """Generate stats and schema for raw ASR data."""
    logging.info("=" * 60)
    logging.info("TFDV: Validating RAW ASR data")
    logging.info("=" * 60)

    raw_path = os.path.join(project_root, "data", "raw", "librispeech", "train_clean_100.parquet")
    if not os.path.exists(raw_path):
        logging.error(f"Raw ASR file not found: {raw_path}")
        return None, None

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(raw_path)
    all_cols = pf.schema.names
    logging.info(f"Raw ASR columns available: {all_cols}")

    light_cols = [c for c in all_cols if c not in ("audio_array",)]
    logging.info(f"Loading raw ASR metadata columns (first 500 rows): {light_cols}")
    batch = next(pf.iter_batches(batch_size=500, columns=light_cols))
    sample_df = batch.to_pandas()
    logging.info(f"Loaded {len(sample_df)} rows, columns={list(sample_df.columns)}")

    flat = flatten_raw_asr(sample_df)
    logging.info(f"Flattened features: {list(flat.columns)}")

    stats = tfdv.generate_statistics_from_dataframe(flat)

    stats_path = os.path.join(output_dir, "raw_asr_stats.pb")
    tfdv.write_stats_text(stats, stats_path)
    logging.info(f"Raw ASR statistics saved to {stats_path}")

    schema = tfdv.infer_schema(stats)

    if "sampling_rate" in flat.columns:
        tfdv.set_domain(schema, "sampling_rate", schema_pb2.FloatDomain(min=8000, max=48000))

    schema_path = os.path.join(output_dir, "raw_asr_schema.pbtxt")
    tfdv.write_schema_text(schema, schema_path)
    logging.info(f"Raw ASR schema saved to {schema_path}")

    return stats, schema


def validate_processed_asr(project_root, output_dir, raw_schema=None):
    """Validate processed ASR data and check for drift against raw schema."""
    logging.info("=" * 60)
    logging.info("TFDV: Validating PROCESSED ASR data")
    logging.info("=" * 60)

    processed_path = os.path.join(project_root, "data", "processed", "asr_processed.parquet")
    if not os.path.exists(processed_path):
        logging.error(f"Processed ASR file not found: {processed_path}")
        return None

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(processed_path)
    all_cols = pf.schema.names
    logging.info(f"Processed ASR columns available: {all_cols}")

    light_cols = [c for c in all_cols if c not in ("log_mel_spec",)]
    logging.info(f"Loading processed ASR metadata columns (first 500 rows): {light_cols}")
    batch = next(pf.iter_batches(batch_size=500, columns=light_cols))
    sample_df = batch.to_pandas()
    logging.info(f"Loaded {len(sample_df)} rows, columns={list(sample_df.columns)}")

    flat = flatten_processed_asr(sample_df)
    logging.info(f"Flattened features: {list(flat.columns)}")

    stats = tfdv.generate_statistics_from_dataframe(flat)

    stats_path = os.path.join(output_dir, "processed_asr_stats.pb")
    tfdv.write_stats_text(stats, stats_path)
    logging.info(f"Processed ASR statistics saved to {stats_path}")

    processed_schema = tfdv.infer_schema(stats)
    schema_path = os.path.join(output_dir, "processed_asr_schema.pbtxt")
    tfdv.write_schema_text(processed_schema, schema_path)
    logging.info(f"Processed ASR schema saved to {schema_path}")

    schema_to_check = raw_schema if raw_schema is not None else processed_schema
    label = "raw" if raw_schema is not None else "self"
    logging.info(f"Checking for anomalies against {label} ASR schema...")
    anomalies = tfdv.validate_statistics(stats, schema_to_check)
    summary = save_anomalies_report(
        anomalies, os.path.join(output_dir, "asr_anomalies")
    )
    if summary["anomaly_count"] > 0:
        logging.warning(
            f"TFDV detected {summary['anomaly_count']} anomalies in processed ASR data!"
        )
        for a in summary["features_with_anomalies"]:
            logging.warning(f"  - {a['feature']}: {a['short_description']}")
    else:
        logging.info("No anomalies detected in processed ASR data.")

    compute_spectrogram_stats(processed_path, output_dir)

    return stats


def main():
    project_root = get_project_root()
    output_dir = os.path.join(project_root, "data", "validation", "asr")
    os.makedirs(output_dir, exist_ok=True)

    raw_stats, raw_schema = validate_raw_asr(project_root, output_dir)
    validate_processed_asr(project_root, output_dir, raw_schema=raw_schema)

    logging.info("ASR TFDV validation complete.")


if __name__ == "__main__":
    main()

"""
Data Slicing for Bias Detection and Mitigation.

Implements SliceFinder-style data slicing to identify demographic or categorical
subgroups, analyze distribution across slices, detect representation bias,
and recommend mitigation strategies. Designed for use with Fairlearn/TFMA
when model evaluation is available; for now performs data-level analysis.

Reference: SliceFinder, TensorFlow Model Analysis (TFMA), Fairlearn
"""

import os
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


UNDERREPRESENTED_THRESHOLD = 0.05  # 5%
MAX_SKEW_RATIO = 10.0


@dataclass
class SliceStats:
    """Statistics for a single data slice."""

    slice_name: str
    slice_key: str
    count: int
    fraction: float
    metrics: Dict[str, float]
    is_underrepresented: bool
    is_overrepresented: bool


@dataclass
class BiasReport:
    """Full bias analysis report for a dataset."""

    dataset_name: str
    total_samples: int
    slice_dimension: str
    slices: List[SliceStats]
    imbalance_detected: bool
    skew_ratio: Optional[float]
    mitigation_recommendations: List[str]
    bias_types_found: List[str]


def slice_nmt_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Prepare NMT data for slicing. Adds slice dimensions: domain, length_bin.
    Uses domain if present; otherwise falls back to length_bin.
    """
    df = df.copy()

    if "domain" not in df.columns:
        df["domain"] = "general"
    df["domain"] = df["domain"].fillna("general").astype(str).str.lower()

    en_wc = df["en"].str.split().str.len() if "en" in df.columns else pd.Series(0, index=df.index)
    df["length_bin"] = pd.cut(
        en_wc.fillna(0),
        bins=[0, 10, 30, np.inf],
        labels=["short", "medium", "long"],
    ).astype(str)

    return df, "domain"


def slice_asr_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Prepare ASR data for slicing. Slice dimensions: text_length_bin (primary),
    speaker_id, chapter_id available for multi-dimensional analysis.
    """
    df = df.copy()

    if "text" in df.columns:
        wc = df["text"].str.split().str.len().fillna(0)
    else:
        wc = pd.Series(10, index=df.index)
    df["text_length_bin"] = pd.cut(
        wc,
        bins=[0, 5, 15, np.inf],
        labels=["short", "medium", "long"],
    ).astype(str)

    if "speaker_id" not in df.columns:
        df["speaker_id"] = "unknown"
    if "chapter_id" not in df.columns:
        df["chapter_id"] = "unknown"

    return df, "text_length_bin"


def compute_slice_statistics(
    df: pd.DataFrame,
    slice_col: str,
    metric_cols: Optional[List[str]] = None,
) -> List[SliceStats]:
    """Compute per-slice counts, fractions, and metrics."""
    if slice_col not in df.columns:
        logging.warning(f"Slice column '{slice_col}' not found. Columns: {list(df.columns)}")
        return []

    total = len(df)
    if total == 0:
        return []

    slice_counts = df[slice_col].value_counts()
    fractions = slice_counts / total
    avg_frac = 1.0 / len(slice_counts) if slice_counts.size > 0 else 0
    underrepresented_thresh = min(UNDERREPRESENTED_THRESHOLD, avg_frac * 0.5)
    overrepresented_thresh = max(1 - UNDERREPRESENTED_THRESHOLD, avg_frac * 2)

    results = []
    for key, count in slice_counts.items():
        frac = fractions[key]
        metrics = {}
        if metric_cols:
            for mc in metric_cols:
                if mc in df.columns:
                    metrics[mc] = float(df.loc[df[slice_col] == key, mc].mean())

        results.append(
            SliceStats(
                slice_name=str(key),
                slice_key=str(key),
                count=int(count),
                fraction=float(frac),
                metrics=metrics,
                is_underrepresented=frac < underrepresented_thresh,
                is_overrepresented=frac > overrepresented_thresh,
            )
        )

    return results


def detect_imbalance(slices: List[SliceStats]) -> Tuple[bool, Optional[float], List[str]]:
    """Detect representation imbalance and bias types."""
    if len(slices) < 2:
        return False, None, []

    counts = [s.count for s in slices]
    skew = max(counts) / min(counts) if min(counts) > 0 else float("inf")
    imbalance = skew > MAX_SKEW_RATIO

    bias_types = []
    under = [s for s in slices if s.is_underrepresented]
    over = [s for s in slices if s.is_overrepresented]
    if under:
        bias_types.append(
            f"Representation bias: {len(under)} slice(s) underrepresented (< {UNDERREPRESENTED_THRESHOLD*100:.0f}%)"
        )
    if over:
        bias_types.append(f"Dominance bias: {len(over)} slice(s) overrepresented")
    if skew > MAX_SKEW_RATIO:
        bias_types.append(
            f"Skew bias: slice size ratio {skew:.1f}x exceeds threshold {MAX_SKEW_RATIO}"
        )

    return imbalance, skew if not np.isinf(skew) else None, bias_types


def recommend_mitigation(
    slices: List[SliceStats],
    dataset_name: str,
    bias_types: List[str],
) -> List[str]:
    """Generate mitigation recommendations based on bias analysis."""
    recs = []

    if bias_types:
        recs.append(f"Detected bias in {dataset_name}. Types: {'; '.join(bias_types)}")

    under = [s for s in slices if s.is_underrepresented]
    if under:
        names = [s.slice_name for s in under]
        recs.append(
            f"Re-sampling: Oversample underrepresented slice(s) [{', '.join(names)}] "
            "during training to balance representation."
        )
        recs.append(
            "Alternative: Use importance weighting (sample_weight) inversely proportional "
            "to slice frequency when training."
        )

    over = [s for s in slices if s.is_overrepresented]
    if over:
        recs.append(
            f"Re-sampling: Consider undersampling overrepresented slice(s) [{', '.join(s.slice_name for s in over)}] "
            "or collecting more data for minority slices."
        )

    recs.append(
        "Fairness constraints: Once a model is trained, use Fairlearn or TFMA to evaluate "
        "performance per slice and apply post-processing (e.g., threshold tuning) if needed."
    )

    recs.append(
        "Document trade-offs: If mitigation reduces overall accuracy, document the "
        "fairness–accuracy trade-off and stakeholder approval."
    )

    return recs


def run_nmt_bias_analysis(project_root: str) -> BiasReport:
    """Run bias analysis on processed NMT data."""
    path = os.path.join(project_root, "data", "processed", "nmt_processed.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"NMT processed file not found: {path}")

    df = pd.read_parquet(path)
    df_sliced, slice_col = slice_nmt_data(df)

    df_sliced["en_word_count"] = df_sliced["en"].str.split().str.len()
    df_sliced["es_word_count"] = df_sliced["es"].str.split().str.len()
    metric_cols = ["en_word_count", "es_word_count"]
    if "en_tokens" in df_sliced.columns:
        df_sliced["en_token_count"] = df_sliced["en_tokens"].apply(
            lambda x: len(x) if isinstance(x, (list, tuple)) else 0
        )
        metric_cols.append("en_token_count")
    if "es_tokens" in df_sliced.columns:
        df_sliced["es_token_count"] = df_sliced["es_tokens"].apply(
            lambda x: len(x) if isinstance(x, (list, tuple)) else 0
        )
        metric_cols.append("es_token_count")

    slices = compute_slice_statistics(df_sliced, slice_col, metric_cols)
    imbalance, skew, bias_types = detect_imbalance(slices)
    recs = recommend_mitigation(slices, "NMT", bias_types)

    return BiasReport(
        dataset_name="NMT (en-es)",
        total_samples=len(df),
        slice_dimension=slice_col,
        slices=slices,
        imbalance_detected=imbalance,
        skew_ratio=skew,
        mitigation_recommendations=recs,
        bias_types_found=bias_types,
    )


def run_asr_bias_analysis(project_root: str) -> BiasReport:
    """Run bias analysis on processed ASR data."""
    path = os.path.join(project_root, "data", "processed", "asr_processed.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ASR processed file not found: {path}")

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    light_cols = [c for c in pf.schema.names if c != "log_mel_spec"]
    df = next(pf.iter_batches(batch_size=5000, columns=light_cols)).to_pandas()

    df_sliced, slice_col = slice_asr_data(df)
    df_sliced["text_word_count"] = (
        df_sliced["text"].str.split().str.len() if "text" in df_sliced.columns else 0
    )

    slices = compute_slice_statistics(df_sliced, slice_col, ["text_word_count"])
    imbalance, skew, bias_types = detect_imbalance(slices)
    recs = recommend_mitigation(slices, "ASR", bias_types)

    return BiasReport(
        dataset_name="ASR (LibriSpeech)",
        total_samples=len(df),
        slice_dimension=slice_col,
        slices=slices,
        imbalance_detected=imbalance,
        skew_ratio=skew,
        mitigation_recommendations=recs,
        bias_types_found=bias_types,
    )


def _to_native(obj: Any) -> Any:
    """Convert numpy/pandas types to native Python for JSON serialization."""
    if isinstance(obj, (np.bool_, np.integer)):
        return int(obj) if isinstance(obj, np.integer) else bool(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    return obj


def report_to_dict(report: BiasReport) -> Dict[str, Any]:
    """Convert BiasReport to JSON-serializable dict."""
    raw = {
        "dataset_name": report.dataset_name,
        "total_samples": report.total_samples,
        "slice_dimension": report.slice_dimension,
        "slices": [
            {
                "slice_name": s.slice_name,
                "count": s.count,
                "fraction": round(float(s.fraction), 4),
                "metrics": s.metrics,
                "is_underrepresented": bool(s.is_underrepresented),
                "is_overrepresented": bool(s.is_overrepresented),
            }
            for s in report.slices
        ],
        "imbalance_detected": bool(report.imbalance_detected),
        "skew_ratio": float(report.skew_ratio) if report.skew_ratio is not None else None,
        "bias_types_found": report.bias_types_found,
        "mitigation_recommendations": report.mitigation_recommendations,
    }
    return _to_native(raw)


def save_report(report: BiasReport, output_dir: str, basename: str) -> str:
    """Save BiasReport as JSON. Returns path to saved file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{basename}.json")
    with open(path, "w") as f:
        json.dump(report_to_dict(report), f, indent=2)
    return path

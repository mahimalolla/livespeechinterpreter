"""
Run bias analysis (data slicing) for ASR and NMT pipelines.

Produces per-slice statistics, imbalance detection, and mitigation
recommendations. Outputs saved to data/validation/bias/.
"""

import os
import sys
import logging

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from data_slicing import (
    run_nmt_bias_analysis,
    run_asr_bias_analysis,
    save_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "..", ".."))


def main():
    project_root = get_project_root()
    output_dir = os.path.join(project_root, "data", "validation", "bias")
    os.makedirs(output_dir, exist_ok=True)

    # NMT bias analysis
    logging.info("=" * 60)
    logging.info("Bias Analysis: NMT (en-es)")
    logging.info("=" * 60)
    try:
        nmt_report = run_nmt_bias_analysis(project_root)
        path = save_report(nmt_report, output_dir, "nmt_bias_report")
        logging.info(f"NMT bias report saved to {path}")
        logging.info(f"  Total samples: {nmt_report.total_samples}")
        logging.info(f"  Slice dimension: {nmt_report.slice_dimension}")
        for s in nmt_report.slices:
            logging.info(
                f"    {s.slice_name}: {s.count} ({s.fraction*100:.1f}%) "
                f"{'(underrepresented)' if s.is_underrepresented else ''}"
            )
        if nmt_report.imbalance_detected:
            logging.warning(f"  Imbalance detected (skew: {nmt_report.skew_ratio:.1f}x)")
        else:
            logging.info("  No significant imbalance detected.")
    except (FileNotFoundError, OSError) as e:
        logging.warning(f"NMT bias analysis skipped: {e}")

    # ASR bias analysis
    logging.info("=" * 60)
    logging.info("Bias Analysis: ASR (LibriSpeech)")
    logging.info("=" * 60)
    try:
        asr_report = run_asr_bias_analysis(project_root)
        path = save_report(asr_report, output_dir, "asr_bias_report")
        logging.info(f"ASR bias report saved to {path}")
        logging.info(f"  Total samples: {asr_report.total_samples}")
        logging.info(f"  Slice dimension: {asr_report.slice_dimension}")
        for s in asr_report.slices:
            logging.info(
                f"    {s.slice_name}: {s.count} ({s.fraction*100:.1f}%) "
                f"{'(underrepresented)' if s.is_underrepresented else ''}"
            )
        if asr_report.imbalance_detected:
            logging.warning(f"  Imbalance detected (skew: {asr_report.skew_ratio:.1f}x)")
        else:
            logging.info("  No significant imbalance detected.")
    except (FileNotFoundError, OSError) as e:
        logging.warning(f"ASR bias analysis skipped: {e}")

    logging.info("Bias analysis complete.")


if __name__ == "__main__":
    main()

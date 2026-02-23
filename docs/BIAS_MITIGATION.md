# Data Bias Detection and Mitigation

This document describes the data slicing approach for bias detection and the mitigation process used in the Live Speech Interpreter pipeline.

---

## 1. Detecting Bias in Your Data

To ensure the data is not biased, we perform **data slicing** and analyze distribution across subgroups. We identify categorical or derived features and evaluate representation and imbalance.

### Identified Slice Dimensions

| Dataset | Slice Dimension | Description |
|---------|-----------------|-------------|
| **NMT (en-es)** | `domain` | general (OPUS), medical, federal |
| **NMT** | `length_bin` | short (1–10 words), medium (11–30), long (31+) |
| **ASR (LibriSpeech)** | `text_length_bin` | short (1–5 words), medium (6–15), long (16+) |
| **ASR** | `speaker_id` | Per-speaker (available for multi-dimensional analysis) |
| **ASR** | `chapter_id` | Per-chapter (book-level variation) |

### How We Detect Bias

1. **Representation bias**: Slices with < 5% of total data are flagged as underrepresented.
2. **Dominance bias**: Slices with > 95% of data are flagged as overrepresented.
3. **Skew bias**: If the ratio (largest slice / smallest slice) exceeds 10×, we flag overall imbalance.

---

## 2. Data Slicing Implementation

We implement SliceFinder-style data slicing using a custom module (`scripts/bias/data_slicing.py`) rather than external tools because:

- **SliceFinder/Sliceline** and **TFMA** require a trained model and error vectors.
- **Fairlearn** is used for model-level fairness metrics.
- Our pipeline focuses on **data-level** analysis before model training.

The implementation:

1. Splits data into meaningful slices by categorical features.
2. Computes per-slice statistics (count, fraction, mean metrics).
3. Flags underrepresented and overrepresented slices.
4. Produces a JSON report suitable for downstream Fairlearn/TFMA integration once a model exists.

### Usage

```bash
python scripts/bias/run_bias_analysis.py
```

Outputs: `data/validation/bias/nmt_bias_report.json`, `data/validation/bias/asr_bias_report.json`

---

## 3. Mitigation of Bias

If significant performance or representation differences are detected:

### Re-sampling Underrepresented Groups

- **Oversample** slices below the 5% threshold during training.
- Use `sample_weight` inversely proportional to slice frequency.

### Re-sampling Overrepresented Groups

- **Undersample** dominant slices or collect more data for minority slices.

### Fairness Constraints (Model-Level)

Once a model is trained:

- Use **Fairlearn** to compute parity metrics (demographic parity, equalized odds) per slice.
- Use **TFMA** to evaluate model performance on each slice.
- Apply **post-processing** (e.g., threshold tuning per slice) to improve fairness.

### Documented Trade-offs

- Mitigation may reduce overall accuracy. Document the fairness–accuracy trade-off.
- Obtain stakeholder approval for any threshold or sampling changes.

---

## 4. Documented Bias Mitigation Process

### Steps Taken

1. **Identify slice dimensions**: Domain (NMT), text length (ASR), speaker/chapter (ASR).
2. **Implement slicing**: `data_slicing.py` with configurable thresholds.
3. **Run analysis**: `run_bias_analysis.py` integrated into the Airflow DAG.
4. **Review reports**: Check `data/validation/bias/*.json` for imbalance flags.
5. **Apply mitigation**: Re-run preprocessing with re-sampling if needed; document model-level mitigation when training begins.

### Types of Bias Found (Example)

- **Representation bias**: Medical/federal domains may be much smaller than general OPUS.
- **Skew bias**: Short utterances (ASR) or short sentences (NMT) may dominate.

### Trade-offs

- Oversampling minority slices can increase training time and risk overfitting to rare slices.
- Undersampling majority slices can reduce overall data quality.
- We recommend starting with importance weighting rather than aggressive resampling.

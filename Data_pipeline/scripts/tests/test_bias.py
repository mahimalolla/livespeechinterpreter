"""Bias slicing tests for data slicing and bias detection."""

import os
import sys
import unittest
import tempfile
import json

import pandas as pd
import numpy as np

_bias_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bias")
sys.path.insert(0, os.path.abspath(_bias_dir))

from data_slicing import (
    slice_nmt_data,
    slice_asr_data,
    compute_slice_statistics,
    detect_imbalance,
    recommend_mitigation,
    report_to_dict,
    save_report,
    _to_native,
    SliceStats,
    BiasReport,
)


class TestSliceNMT(unittest.TestCase):
    def test_slice_nmt_domain(self):
        df = pd.DataFrame({
            "en": ["Hello world"] * 80 + ["Patient needs treatment"] * 15 + ["Federal law"] * 5,
            "es": ["Hola mundo"] * 80 + ["Paciente necesita tratamiento"] * 15 + ["Ley federal"] * 5,
        })
        df["domain"] = ["general"] * 80 + ["medical"] * 15 + ["federal"] * 5
        df_sliced, col = slice_nmt_data(df)
        self.assertEqual(col, "domain")
        self.assertIn("length_bin", df_sliced.columns)
        self.assertIn("domain", df_sliced.columns)

    def test_slice_nmt_no_domain(self):
        df = pd.DataFrame({
            "en": ["Hello world"] * 10,
            "es": ["Hola mundo"] * 10,
        })
        df_sliced, col = slice_nmt_data(df)
        self.assertEqual(col, "domain")
        self.assertTrue((df_sliced["domain"] == "general").all())


class TestSliceASR(unittest.TestCase):
    def test_slice_asr_text_length(self):
        df = pd.DataFrame({
            "text": ["a b"] * 30 + ["one two three four five six"] * 50 + ["x"] * 20,
            "speaker_id": ["s1"] * 50 + ["s2"] * 50,
            "chapter_id": ["c1"] * 100,
        })
        df_sliced, col = slice_asr_data(df)
        self.assertEqual(col, "text_length_bin")
        self.assertIn("text_length_bin", df_sliced.columns)
        counts = df_sliced["text_length_bin"].value_counts()
        self.assertGreaterEqual(len(counts), 2)


class TestImbalance(unittest.TestCase):
    def test_balanced(self):
        slices = [
            SliceStats("a", "a", 33, 0.33, {}, False, False),
            SliceStats("b", "b", 33, 0.33, {}, False, False),
            SliceStats("c", "c", 34, 0.34, {}, False, False),
        ]
        imb, skew, bias = detect_imbalance(slices)
        self.assertFalse(imb)
        self.assertLess(skew or 0, 10)

    def test_imbalanced(self):
        slices = [
            SliceStats("a", "a", 95, 0.95, {}, False, True),
            SliceStats("b", "b", 5, 0.05, {}, True, False),
        ]
        imb, skew, bias = detect_imbalance(slices)
        self.assertTrue(imb)
        self.assertGreater(skew or 0, 10)
        self.assertGreater(len(bias), 0)


class TestMitigation(unittest.TestCase):
    def test_recommendations(self):
        slices = [
            SliceStats("medical", "medical", 2, 0.02, {}, True, False),
            SliceStats("general", "general", 98, 0.98, {}, False, True),
        ]
        recs = recommend_mitigation(slices, "NMT", ["Representation bias"])
        self.assertGreater(len(recs), 0)
        self.assertTrue(any("oversample" in r.lower() or "re-sampl" in r.lower() for r in recs))


class TestComputeSliceStatistics(unittest.TestCase):
    """Tests for compute_slice_statistics."""

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["domain"])
        result = compute_slice_statistics(df, "domain")
        self.assertEqual(result, [])

    def test_missing_slice_column(self):
        df = pd.DataFrame({"en": ["a"], "es": ["b"]})
        result = compute_slice_statistics(df, "domain")
        self.assertEqual(result, [])

    def test_with_metrics(self):
        df = pd.DataFrame({
            "domain": ["general", "general", "medical"],
            "en_word_count": [5, 10, 15],
        })
        result = compute_slice_statistics(df, "domain", ["en_word_count"])
        self.assertEqual(len(result), 2)
        gen = next(s for s in result if s.slice_name == "general")
        self.assertEqual(gen.count, 2)
        self.assertIn("en_word_count", gen.metrics)
        self.assertEqual(gen.metrics["en_word_count"], 7.5)


class TestReportSerialization(unittest.TestCase):
    """Tests for report_to_dict and save_report (JSON serialization)."""

    def test_report_to_dict_json_serializable(self):
        report = BiasReport(
            dataset_name="NMT",
            total_samples=100,
            slice_dimension="domain",
            slices=[
                SliceStats("general", "general", 80, 0.8, {"en_word_count": 12.5}, False, True),
                SliceStats("medical", "medical", 20, 0.2, {"en_word_count": 15.0}, False, False),
            ],
            imbalance_detected=False,
            skew_ratio=4.0,
            mitigation_recommendations=["Doc trade-offs"],
            bias_types_found=[],
        )
        d = report_to_dict(report)
        # Must be JSON serializable (no numpy types)
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["dataset_name"], "NMT")
        self.assertEqual(parsed["total_samples"], 100)
        self.assertEqual(len(parsed["slices"]), 2)

    def test_save_report_creates_file(self):
        report = BiasReport(
            dataset_name="Test",
            total_samples=10,
            slice_dimension="domain",
            slices=[SliceStats("a", "a", 10, 1.0, {}, False, False)],
            imbalance_detected=False,
            skew_ratio=None,
            mitigation_recommendations=[],
            bias_types_found=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = save_report(report, tmp, "test_report")
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["dataset_name"], "Test")


class TestToNative(unittest.TestCase):
    """Tests for _to_native (numpy to Python conversion)."""

    def test_numpy_bool(self):
        self.assertEqual(_to_native(np.bool_(True)), True)
        self.assertEqual(_to_native(np.bool_(False)), False)

    def test_numpy_float(self):
        self.assertEqual(_to_native(np.float64(3.14)), 3.14)

    def test_dict_recursive(self):
        d = {"a": np.bool_(True), "b": [np.float64(1.0)]}
        out = _to_native(d)
        self.assertEqual(out["a"], True)
        self.assertEqual(out["b"][0], 1.0)


if __name__ == "__main__":
    unittest.main()

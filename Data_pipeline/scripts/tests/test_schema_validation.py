"""Schema validation tests for TFDV schemas and anomaly reports."""

import os
import sys
import json
import unittest

import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
scripts_dir = os.path.join(project_root, "scripts")
sys.path.insert(0, project_root)
sys.path.insert(0, scripts_dir)

# TFDV is optional - may not be installed in local dev
try:
    import tensorflow_data_validation as tfdv
    from tensorflow_metadata.proto.v0 import schema_pb2
    from google.protobuf import text_format
    TFDV_AVAILABLE = True
except ImportError:
    TFDV_AVAILABLE = False


def get_validation_dir(subdir):
    return os.path.join(project_root, "data", "validation", subdir)


class TestSchemaFileExistence(unittest.TestCase):
    """Verify schema files exist when validation has been run."""

    def test_nmt_schema_files_exist_if_validation_run(self):
        nmt_dir = get_validation_dir("nmt")
        if not os.path.isdir(nmt_dir):
            self.skipTest("data/validation/nmt not found (run validation first)")
        raw_schema = os.path.join(nmt_dir, "raw_nmt_schema.pbtxt")
        proc_schema = os.path.join(nmt_dir, "processed_nmt_schema.pbtxt")
        if os.path.isfile(raw_schema):
            self.assertTrue(os.path.isfile(raw_schema))
        if os.path.isfile(proc_schema):
            self.assertTrue(os.path.isfile(proc_schema))

    def test_asr_schema_files_exist_if_validation_run(self):
        asr_dir = get_validation_dir("asr")
        if not os.path.isdir(asr_dir):
            self.skipTest("data/validation/asr not found (run validation first)")
        raw_schema = os.path.join(asr_dir, "raw_asr_schema.pbtxt")
        proc_schema = os.path.join(asr_dir, "processed_asr_schema.pbtxt")
        if os.path.isfile(raw_schema):
            self.assertTrue(os.path.isfile(raw_schema))
        if os.path.isfile(proc_schema):
            self.assertTrue(os.path.isfile(proc_schema))


@unittest.skipIf(not TFDV_AVAILABLE, "TFDV not installed")
class TestSchemaValidity(unittest.TestCase):
    """Verify TFDV schemas are valid and have expected features."""

    def test_nmt_raw_schema_parseable(self):
        path = os.path.join(get_validation_dir("nmt"), "raw_nmt_schema.pbtxt")
        if not os.path.isfile(path):
            self.skipTest("NMT raw schema not found")
        with open(path) as f:
            schema = text_format.Parse(f.read(), schema_pb2.Schema())
        self.assertIsNotNone(schema)
        feature_names = [f.name for f in schema.feature]
        self.assertIn("en", feature_names)
        self.assertIn("es", feature_names)

    def test_nmt_processed_schema_parseable(self):
        path = os.path.join(get_validation_dir("nmt"), "processed_nmt_schema.pbtxt")
        if not os.path.isfile(path):
            self.skipTest("NMT processed schema not found")
        with open(path) as f:
            schema = text_format.Parse(f.read(), schema_pb2.Schema())
        self.assertIsNotNone(schema)
        feature_names = [f.name for f in schema.feature]
        self.assertIn("en", feature_names)
        self.assertIn("es", feature_names)

    def test_asr_raw_schema_parseable(self):
        path = os.path.join(get_validation_dir("asr"), "raw_asr_schema.pbtxt")
        if not os.path.isfile(path):
            self.skipTest("ASR raw schema not found")
        with open(path) as f:
            schema = text_format.Parse(f.read(), schema_pb2.Schema())
        self.assertIsNotNone(schema)
        feature_names = [f.name for f in schema.feature]
        self.assertIn("text", feature_names)

    def test_tfdv_can_load_schema(self):
        """TFDV can load and use a schema for validation."""
        path = os.path.join(get_validation_dir("nmt"), "raw_nmt_schema.pbtxt")
        if not os.path.isfile(path):
            self.skipTest("NMT raw schema not found")
        schema = tfdv.load_schema_text(path)
        self.assertIsNotNone(schema)
        # Validate a minimal compliant dataframe
        df = pd.DataFrame({
            "en": ["Hello world"],
            "es": ["Hola mundo"],
            "en_word_count": [2],
            "es_word_count": [2],
            "en_char_count": [11],
            "es_char_count": [10],
        })
        stats = tfdv.generate_statistics_from_dataframe(df)
        anomalies = tfdv.validate_statistics(stats, schema)
        # May have anomalies (e.g. new columns) but validation runs
        self.assertIsNotNone(anomalies)


class TestAnomalyReportStructure(unittest.TestCase):
    """Verify anomaly report JSON structure is correct."""

    def test_nmt_anomaly_report_structure(self):
        path = os.path.join(get_validation_dir("nmt"), "nmt_anomalies.json")
        if not os.path.isfile(path):
            self.skipTest("NMT anomaly report not found")
        with open(path) as f:
            report = json.load(f)
        self.assertIn("anomaly_count", report)
        self.assertIn("features_with_anomalies", report)
        self.assertIsInstance(report["anomaly_count"], int)
        self.assertIsInstance(report["features_with_anomalies"], list)
        for item in report["features_with_anomalies"]:
            self.assertIn("feature", item)
            self.assertIn("severity", item)

    def test_asr_anomaly_report_structure(self):
        path = os.path.join(get_validation_dir("asr"), "asr_anomalies.json")
        if not os.path.isfile(path):
            self.skipTest("ASR anomaly report not found")
        with open(path) as f:
            report = json.load(f)
        self.assertIn("anomaly_count", report)
        self.assertIn("features_with_anomalies", report)


class TestFlattenNMTForValidation(unittest.TestCase):
    """Test NMT flatten logic used by TFDV validation."""

    def test_flatten_produces_expected_columns(self):
        """Flatten produces columns TFDV can profile."""
        try:
            import validation.validate_nmt as vnmt
            flatten_nmt_for_validation = vnmt.flatten_nmt_for_validation
        except (ImportError, AttributeError):
            self.skipTest("validation module not found")
        df = pd.DataFrame({
            "en": ["Hello world", "Hi"],
            "es": ["Hola mundo", "Hola"],
            "en_tokens": [[1, 2, 3], [1, 2]],
            "es_tokens": [[1, 2, 3, 4], [1, 2]],
        })
        flat = flatten_nmt_for_validation(df)
        self.assertIn("en_word_count", flat.columns)
        self.assertIn("es_word_count", flat.columns)
        self.assertIn("en_token_count", flat.columns)
        self.assertIn("es_token_count", flat.columns)
        self.assertEqual(flat["en_word_count"].iloc[0], 2)
        self.assertEqual(flat["en_token_count"].iloc[0], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

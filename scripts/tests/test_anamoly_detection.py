
import json
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

import anomaly_detection as ad


def base_record(**overrides):
    r = {
        "instruction": "Translate",
        "input": "The patient has a fever and needs treatment now.",
        "output": "El paciente tiene fiebre y necesita tratamiento ahora.",
        "domain": "medical",
        "direction": "en_to_es",
    }
    r.update(overrides)
    return r


def make_val_jsonl(records):
    return "\n".join(json.dumps(r) for r in records) + "\n"


class TestDetectAnomalies:

    @patch.object(ad, "json")
    @patch.object(ad, "os")
    @patch("builtins.open")
    def test_no_anomalies_in_clean_data(self, mock_open_fn, mock_os, mock_json):
        records = [base_record() for _ in range(20)]
        jsonl = make_val_jsonl(records)

        # Make json.loads work normally for parsing JSONL lines
        mock_json.loads = json.loads
        mock_json.dump = MagicMock()

        def open_side_effect(path, *args, **kwargs):
            if "val.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(jsonl), __exit__=lambda *a: None)
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        mock_open_fn.side_effect = open_side_effect
        ad.detect_anomalies()

        report = mock_json.dump.call_args[0][0]
        assert report["anomalies"] == []
        assert report["total_checked"] == 20

    @patch.object(ad, "json")
    @patch.object(ad, "os")
    @patch("builtins.open")
    def test_detects_too_long_input(self, mock_open_fn, mock_os, mock_json):
        long_input = " ".join(["word"] * 250)
        records = [base_record(input=long_input)]
        jsonl = make_val_jsonl(records)

        mock_json.loads = json.loads
        mock_json.dump = MagicMock()

        def open_side_effect(path, *args, **kwargs):
            if "val.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(jsonl), __exit__=lambda *a: None)
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        mock_open_fn.side_effect = open_side_effect

        with pytest.raises(ValueError, match="anomalies detected"):
            ad.detect_anomalies()

    @patch.object(ad, "json")
    @patch.object(ad, "os")
    @patch("builtins.open")
    def test_detects_too_short_input(self, mock_open_fn, mock_os, mock_json):
        records = [base_record(input="Hi")]
        jsonl = make_val_jsonl(records)

        mock_json.loads = json.loads
        mock_json.dump = MagicMock()

        def open_side_effect(path, *args, **kwargs):
            if "val.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(jsonl), __exit__=lambda *a: None)
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        mock_open_fn.side_effect = open_side_effect

        with pytest.raises(ValueError):
            ad.detect_anomalies()

    @patch.object(ad, "json")
    @patch.object(ad, "os")
    @patch("builtins.open")
    def test_detects_invalid_domain(self, mock_open_fn, mock_os, mock_json):
        records = [base_record(domain="finance")]
        jsonl = make_val_jsonl(records)

        mock_json.loads = json.loads
        mock_json.dump = MagicMock()

        def open_side_effect(path, *args, **kwargs):
            if "val.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(jsonl), __exit__=lambda *a: None)
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        mock_open_fn.side_effect = open_side_effect

        with pytest.raises(ValueError, match="anomalies detected"):
            ad.detect_anomalies()

    @patch.object(ad, "json")
    @patch.object(ad, "os")
    @patch("builtins.open")
    def test_detects_invalid_direction(self, mock_open_fn, mock_os, mock_json):
        records = [base_record(direction="fr_to_en")]
        jsonl = make_val_jsonl(records)

        mock_json.loads = json.loads
        mock_json.dump = MagicMock()

        def open_side_effect(path, *args, **kwargs):
            if "val.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(jsonl), __exit__=lambda *a: None)
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        mock_open_fn.side_effect = open_side_effect

        with pytest.raises(ValueError):
            ad.detect_anomalies()

    @patch.object(ad, "json")
    @patch.object(ad, "os")
    @patch("builtins.open")
    def test_detects_null_inputs(self, mock_open_fn, mock_os, mock_json):
        records = [base_record(input="")]
        jsonl = make_val_jsonl(records)

        mock_json.loads = json.loads
        mock_json.dump = MagicMock()

        def open_side_effect(path, *args, **kwargs):
            if "val.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(jsonl), __exit__=lambda *a: None)
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        mock_open_fn.side_effect = open_side_effect

        with pytest.raises(ValueError):
            ad.detect_anomalies()

    @patch.object(ad, "json")
    @patch.object(ad, "os")
    @patch("builtins.open")
    def test_detects_null_outputs(self, mock_open_fn, mock_os, mock_json):
        records = [base_record(output="")]
        jsonl = make_val_jsonl(records)

        mock_json.loads = json.loads
        mock_json.dump = MagicMock()

        def open_side_effect(path, *args, **kwargs):
            if "val.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(jsonl), __exit__=lambda *a: None)
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        mock_open_fn.side_effect = open_side_effect

        with pytest.raises(ValueError):
            ad.detect_anomalies()

    @patch.object(ad, "json")
    @patch.object(ad, "os")
    @patch("builtins.open")
    def test_report_written_even_on_anomaly(self, mock_open_fn, mock_os, mock_json):
        records = [base_record(domain="invalid")]
        jsonl = make_val_jsonl(records)

        mock_json.loads = json.loads
        mock_json.dump = MagicMock()

        def open_side_effect(path, *args, **kwargs):
            if "val.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(jsonl), __exit__=lambda *a: None)
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        mock_open_fn.side_effect = open_side_effect

        with pytest.raises(ValueError):
            ad.detect_anomalies()

        assert mock_json.dump.called
        report = mock_json.dump.call_args[0][0]
        assert len(report["anomalies"]) > 0
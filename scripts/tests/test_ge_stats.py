

import json
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

import ge_stats as gs


def make_train_jsonl(records):
    return "\n".join(json.dumps(r) for r in records) + "\n"


def base_record(**overrides):
    r = {
        "instruction": "Translate",
        "input": "The patient has a fever and cough.",
        "output": "El paciente tiene fiebre y tos.",
        "domain": "medical",
        "direction": "en_to_es",
    }
    r.update(overrides)
    return r


class TestGenerateGeStats:

    @patch.object(gs, "json")
    @patch.object(gs, "os")
    @patch("builtins.open")
    def test_counts_total_records(self, mock_open_fn, mock_os, mock_json):
        records = [base_record() for _ in range(50)]
        jsonl = make_train_jsonl(records)

        def open_side_effect(path, *args, **kwargs):
            if "train.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(jsonl), __exit__=lambda *a: None)
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        mock_open_fn.side_effect = open_side_effect
        gs.generate_ge_stats()

        summary = mock_json.dump.call_args_list[0][0][0]
        assert summary["total_records"] == 50

    @patch.object(gs, "json")
    @patch.object(gs, "os")
    @patch("builtins.open")
    def test_domain_distribution(self, mock_open_fn, mock_os, mock_json):
        records = [base_record(domain="medical")] * 30 + [base_record(domain="legal")] * 20
        jsonl = make_train_jsonl(records)

        def open_side_effect(path, *args, **kwargs):
            if "train.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(jsonl), __exit__=lambda *a: None)
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        mock_open_fn.side_effect = open_side_effect
        gs.generate_ge_stats()

        summary = mock_json.dump.call_args_list[0][0][0]
        assert summary["domain_distribution"]["medical"] == 30
        assert summary["domain_distribution"]["legal"] == 20

    @patch.object(gs, "json")
    @patch.object(gs, "os")
    @patch("builtins.open")
    def test_null_input_counting(self, mock_open_fn, mock_os, mock_json):
        records = [base_record()] * 5 + [base_record(input="")] * 3
        jsonl = make_train_jsonl(records)

        def open_side_effect(path, *args, **kwargs):
            if "train.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(jsonl), __exit__=lambda *a: None)
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        mock_open_fn.side_effect = open_side_effect
        gs.generate_ge_stats()

        summary = mock_json.dump.call_args_list[0][0][0]
        assert summary["null_input_count"] == 3

    @patch.object(gs, "json")
    @patch.object(gs, "os")
    @patch("builtins.open")
    def test_schema_has_required_fields(self, mock_open_fn, mock_os, mock_json):
        records = [base_record() for _ in range(10)]
        jsonl = make_train_jsonl(records)

        def open_side_effect(path, *args, **kwargs):
            if "train.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(jsonl), __exit__=lambda *a: None)
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        mock_open_fn.side_effect = open_side_effect
        gs.generate_ge_stats()

        schema = mock_json.dump.call_args_list[1][0][0]
        assert "instruction" in schema["required_fields"]
        assert "input" in schema["required_fields"]
        assert schema["valid_domains"] == ["medical", "legal"]

    @patch.object(gs, "json")
    @patch.object(gs, "os")
    @patch("builtins.open")
    def test_input_length_stats(self, mock_open_fn, mock_os, mock_json):
        records = [
            base_record(input="one two three"),
            base_record(input="one two three four five"),
            base_record(input="one two three four five six seven"),
        ]
        jsonl = make_train_jsonl(records)

        def open_side_effect(path, *args, **kwargs):
            if "train.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(jsonl), __exit__=lambda *a: None)
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        mock_open_fn.side_effect = open_side_effect
        gs.generate_ge_stats()

        summary = mock_json.dump.call_args_list[0][0][0]
        assert summary["input_len_min"] == 3
        assert summary["input_len_max"] == 7
        assert summary["input_len_mean"] == 5.0
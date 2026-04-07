
import json
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

import dataset_approval as da


def make_jsonl(n):
    return "\n".join([json.dumps({"input": f"line {i}"}) for i in range(n)]) + "\n"


class TestApproveDataset:

    @patch.object(da, "json")
    @patch("builtins.open")
    def test_approves_when_train_exceeds_minimum(self, mock_open_fn, mock_json):
        mock_json.dump = MagicMock()

        def open_side_effect(path, *args, **kwargs):
            if "train.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(make_jsonl(15000)), __exit__=lambda *a: None)
            if "val.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(make_jsonl(2000)), __exit__=lambda *a: None)
            if "test.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(make_jsonl(2000)), __exit__=lambda *a: None)
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        mock_open_fn.side_effect = open_side_effect
        da.approve_dataset()

        approval = mock_json.dump.call_args[0][0]
        assert approval["approved"] is True
        assert approval["train_records"] == 15000
        assert approval["val_records"] == 2000
        assert approval["test_records"] == 2000
        assert approval["version"] == "v2_approved"

    @patch.object(da, "json")
    @patch("builtins.open")
    def test_rejects_when_train_too_small(self, mock_open_fn, mock_json):
        mock_json.dump = MagicMock()

        def open_side_effect(path, *args, **kwargs):
            if "train.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(make_jsonl(500)), __exit__=lambda *a: None)
            if "val.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(make_jsonl(50)), __exit__=lambda *a: None)
            if "test.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(make_jsonl(50)), __exit__=lambda *a: None)
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        mock_open_fn.side_effect = open_side_effect

        with pytest.raises(ValueError, match="Train set too small"):
            da.approve_dataset()

    @patch.object(da, "json")
    @patch("builtins.open")
    def test_rejects_at_exact_boundary(self, mock_open_fn, mock_json):
        mock_json.dump = MagicMock()

        def open_side_effect(path, *args, **kwargs):
            if "train.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(make_jsonl(9999)), __exit__=lambda *a: None)
            if "val.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(make_jsonl(100)), __exit__=lambda *a: None)
            if "test.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(make_jsonl(100)), __exit__=lambda *a: None)
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        mock_open_fn.side_effect = open_side_effect

        with pytest.raises(ValueError):
            da.approve_dataset()

    @patch.object(da, "json")
    @patch("builtins.open")
    def test_approves_at_exact_minimum(self, mock_open_fn, mock_json):
        mock_json.dump = MagicMock()

        def open_side_effect(path, *args, **kwargs):
            if "train.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(make_jsonl(10000)), __exit__=lambda *a: None)
            if "val.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(make_jsonl(100)), __exit__=lambda *a: None)
            if "test.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(make_jsonl(100)), __exit__=lambda *a: None)
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        mock_open_fn.side_effect = open_side_effect
        da.approve_dataset()

        approval = mock_json.dump.call_args[0][0]
        assert approval["approved"] is True
        assert approval["train_records"] == 10000

    @patch.object(da, "json")
    @patch("builtins.open")
    def test_approval_has_timestamp(self, mock_open_fn, mock_json):
        mock_json.dump = MagicMock()

        def open_side_effect(path, *args, **kwargs):
            if "train.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(make_jsonl(10000)), __exit__=lambda *a: None)
            if "val.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(make_jsonl(100)), __exit__=lambda *a: None)
            if "test.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(make_jsonl(100)), __exit__=lambda *a: None)
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        mock_open_fn.side_effect = open_side_effect
        da.approve_dataset()

        approval = mock_json.dump.call_args[0][0]
        assert "timestamp" in approval
        assert len(approval["timestamp"]) > 0

    @patch.object(da, "json")
    @patch("builtins.open")
    def test_skips_empty_lines_in_count(self, mock_open_fn, mock_json):
        mock_json.dump = MagicMock()
        train_data = "\n\n" + make_jsonl(10000) + "\n\n"

        def open_side_effect(path, *args, **kwargs):
            if "train.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(train_data), __exit__=lambda *a: None)
            if "val.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(make_jsonl(100)), __exit__=lambda *a: None)
            if "test.jsonl" in path:
                return MagicMock(__enter__=lambda s: StringIO(make_jsonl(100)), __exit__=lambda *a: None)
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        mock_open_fn.side_effect = open_side_effect
        da.approve_dataset()

        approval = mock_json.dump.call_args[0][0]
        assert approval["train_records"] == 10000
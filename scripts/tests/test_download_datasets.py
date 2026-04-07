

import json
import pytest
from unittest.mock import patch, mock_open, MagicMock

import download_datasets as dd


class TestDownloadDatasets:

    @patch.object(dd, "json")
    @patch.object(dd, "hashlib")
    @patch("builtins.open", mock_open(read_data=b"fake"))
    @patch.object(dd, "load_dataset")
    @patch.object(dd, "os")
    def test_creates_raw_directory(self, mock_os, mock_load, mock_hash, mock_json):
        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=MagicMock())
        mock_load.return_value = mock_ds
        mock_hash.md5.return_value.hexdigest.return_value = "abc"

        dd.download_datasets()
        mock_os.makedirs.assert_called_once_with("/opt/airflow/data/raw", exist_ok=True)

    @patch.object(dd, "json")
    @patch.object(dd, "hashlib")
    @patch("builtins.open", mock_open(read_data=b"fake"))
    @patch.object(dd, "load_dataset")
    @patch.object(dd, "os")
    def test_downloads_both_datasets(self, mock_os, mock_load, mock_hash, mock_json):
        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=MagicMock())
        mock_load.return_value = mock_ds
        mock_hash.md5.return_value.hexdigest.return_value = "abc"

        dd.download_datasets()
        assert mock_load.call_count == 2

    @patch.object(dd, "json")
    @patch.object(dd, "hashlib")
    @patch("builtins.open", mock_open(read_data=b"fake"))
    @patch.object(dd, "load_dataset")
    @patch.object(dd, "os")
    def test_manifest_has_correct_domains(self, mock_os, mock_load, mock_hash, mock_json):
        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=MagicMock())
        mock_load.return_value = mock_ds
        mock_hash.md5.return_value.hexdigest.return_value = "abc"

        dd.download_datasets()

        manifest_arg = mock_json.dump.call_args_list[-1][0][0]
        assert manifest_arg["emea_medical"]["domain"] == "medical"
        assert manifest_arg["europarl_legal"]["domain"] == "legal"

    @patch.object(dd, "json")
    @patch.object(dd, "hashlib")
    @patch("builtins.open", mock_open(read_data=b"fake"))
    @patch.object(dd, "load_dataset")
    @patch.object(dd, "os")
    def test_manifest_includes_checksums(self, mock_os, mock_load, mock_hash, mock_json):
        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=MagicMock())
        mock_load.return_value = mock_ds
        mock_hash.md5.return_value.hexdigest.return_value = "checksum123"

        dd.download_datasets()

        manifest_arg = mock_json.dump.call_args_list[-1][0][0]
        assert manifest_arg["emea_medical"]["checksum"] == "checksum123"

    @patch.object(dd, "json")
    @patch.object(dd, "hashlib")
    @patch("builtins.open", mock_open(read_data=b"fake"))
    @patch.object(dd, "load_dataset")
    @patch.object(dd, "os")
    def test_load_dataset_called_with_correct_configs(self, mock_os, mock_load, mock_hash, mock_json):
        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=MagicMock())
        mock_load.return_value = mock_ds
        mock_hash.md5.return_value.hexdigest.return_value = "abc"

        dd.download_datasets()

        call_args = [c[0] for c in mock_load.call_args_list]
        hf_ids = [a[0] for a in call_args]
        assert "Helsinki-NLP/opus-100" in hf_ids
        assert "Helsinki-NLP/europarl" in hf_ids
import pytest
from unittest.mock import patch, MagicMock

import upload_gcs as ug


class TestUploadToGCS:

    @patch.object(ug, "os")
    @patch.object(ug, "storage")
    def test_uploads_all_expected_files(self, mock_storage, mock_os):
        mock_os.path.exists.return_value = True
        mock_os.path.getsize.return_value = 1048576
        mock_os.listdir.return_value = ["medical.jsonl", "legal.jsonl"]
        mock_os.path.join = lambda *args: "/".join(args)
        mock_os.path.basename = lambda p: p.split("/")[-1]

        mock_client = MagicMock()
        mock_storage.Client.from_service_account_json.return_value = mock_client
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        ug.upload_to_gcs()

        uploaded_blobs = [c[0][0] for c in mock_bucket.blob.call_args_list]
        # 7 base files + 2 slice files
        assert len(uploaded_blobs) == 9

    @patch.object(ug, "os")
    @patch.object(ug, "storage")
    def test_uploads_base_files_when_no_slices(self, mock_storage, mock_os):
        mock_os.path.exists.return_value = True
        mock_os.path.getsize.return_value = 1048576
        mock_os.listdir.return_value = []
        mock_os.path.basename = lambda p: p.split("/")[-1]

        mock_client = MagicMock()
        mock_storage.Client.from_service_account_json.return_value = mock_client
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        ug.upload_to_gcs()

        assert mock_blob.upload_from_filename.call_count == 7

    @patch.object(ug, "os")
    @patch.object(ug, "storage")
    def test_skips_missing_files(self, mock_storage, mock_os):
        existing_files = {
            "/opt/airflow/data/processed/train.jsonl",
            "/opt/airflow/data/processed/val.jsonl",
            "/opt/airflow/data/processed/test.jsonl",
        }
        mock_os.path.exists.side_effect = lambda p: p in existing_files
        mock_os.path.getsize.return_value = 1048576
        mock_os.listdir.return_value = []
        mock_os.path.basename = lambda p: p.split("/")[-1]

        mock_client = MagicMock()
        mock_storage.Client.from_service_account_json.return_value = mock_client
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        ug.upload_to_gcs()

        assert mock_blob.upload_from_filename.call_count == 3

    @patch.object(ug, "os")
    @patch.object(ug, "storage")
    def test_uses_correct_gcs_prefix(self, mock_storage, mock_os):
        mock_os.path.exists.return_value = True
        mock_os.path.getsize.return_value = 1048576
        mock_os.listdir.return_value = []
        mock_os.path.basename = lambda p: p.split("/")[-1]

        mock_client = MagicMock()
        mock_storage.Client.from_service_account_json.return_value = mock_client
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        ug.upload_to_gcs()

        blob_names = [c[0][0] for c in mock_bucket.blob.call_args_list]
        for name in blob_names:
            assert name.startswith("datasets/v2_approved/")

    @patch.object(ug, "os")
    @patch.object(ug, "storage")
    def test_connects_to_correct_bucket(self, mock_storage, mock_os):
        mock_os.path.exists.return_value = True
        mock_os.path.getsize.return_value = 1048576
        mock_os.listdir.return_value = []
        mock_os.path.basename = lambda p: p.split("/")[-1]

        mock_client = MagicMock()
        mock_storage.Client.from_service_account_json.return_value = mock_client
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        ug.upload_to_gcs()

        mock_client.bucket.assert_called_with("livespeechinterpreter")
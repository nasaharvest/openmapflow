import contextlib
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

try:
    import ts  # noqa: F401

    TORCHSERVE_INSTALLED = True
except ImportError:
    TORCHSERVE_INSTALLED = False

if TORCHSERVE_INSTALLED:
    from docker.torchserve_handler import (
        download_file,
        get_bucket_name,
        get_path,
        upload_file,
    )

tempdir = tempfile.gettempdir()


@contextlib.contextmanager
def create_and_delete_temp_file(filename):
    """
    Context manager that creates a temporary file and deletes it when the context is exited.
    """
    f = Path(f"{tempdir}/{filename}")
    f.touch()
    yield str(f)
    f.unlink()


class TestTorchserveHandler(TestCase):
    def setUp(self) -> None:
        if not TORCHSERVE_INSTALLED:
            self.skipTest("Torchserve is not installed")

    def test_get_bucket_name(self):
        self.assertEqual(get_bucket_name("gs://bucket1/path/to/file"), "bucket1")
        self.assertEqual(get_bucket_name("gs://bucket2"), "bucket2")
        self.assertRaises(ValueError, get_bucket_name, "s3://aws-bucket")
        self.assertRaises(ValueError, get_bucket_name, "gs://")

    def test_get_path(self):
        self.assertEqual(get_path("gs://bucket1/path/to/file"), "path/to/file")
        self.assertEqual(get_path("gs://bucket1/file"), "file")
        self.assertRaises(ValueError, get_path, "gs://bucket")

    def test_get_and_replace_path(self):
        self.assertEqual(
            get_path("gs://bucket/path/to/file", replace_filename="file2"),
            "path/to/file2",
        )
        self.assertEqual(
            get_path("gs://bucket/file", replace_filename="file2"),
            "file2",
        )

    @patch("docker.torchserve_handler.storage")
    def test_download_file_failure(self, mock_storage):
        self.assertRaises(
            FileExistsError, download_file, "gs://fake-bucket/fake-file.tif"
        )

    @patch("docker.torchserve_handler.storage")
    def test_download_file_default(self, mock_storage):
        with create_and_delete_temp_file("fake-file.tif") as expected_local_path:
            actual_local_path = download_file("gs://fake-bucket/fake-file.tif")
            self.assertEqual(expected_local_path, actual_local_path)

    @patch("docker.torchserve_handler.storage")
    def test_download_file_nested(self, mock_storage):
        with create_and_delete_temp_file("fake-file.tif") as expected_local_path:
            actual_local_path = download_file(
                "gs://fake-bucket/dir1/dir2/dir3/fake-file.tif"
            )
            self.assertEqual(expected_local_path, actual_local_path)

    @patch("docker.torchserve_handler.storage")
    def test_download_file_gcloud_calls(self, mock_storage):
        bucket = "fake-bucket"
        filename = "fake-file.tif"
        with create_and_delete_temp_file(filename):
            download_file(f"gs://{bucket}/{filename}")

        mock_storage.Client.assert_called()
        mock_storage_client = mock_storage.Client()
        mock_storage_client.bucket.assert_called_once()
        mock_storage_bucket = mock_storage_client.bucket(bucket)
        mock_storage_bucket.blob.assert_called_once()
        mock_storage_bucket.blob(filename).exists.assert_called_once()
        mock_storage_bucket.blob(filename).download_to_filename.assert_called_once()

    @patch("docker.torchserve_handler.storage")
    def test_upload_file(self, mock_storage):
        bucket = "fake-bucket"
        filename = "fake-file.tif"
        src_uri = "gs://original-bucket/original-file.tif"
        with create_and_delete_temp_file(filename) as local_path:
            actual_upload_path = upload_file(
                bucket_name=bucket, local_path=Path(local_path), src_uri=src_uri
            )
            self.assertEqual(actual_upload_path, f"gs://{bucket}/{filename}")

    @patch("docker.torchserve_handler.storage")
    def test_upload_file_nested(self, mock_storage):
        bucket = "fake-bucket"
        filename = "fake-file.tif"
        src_uri = "gs://original-bucket/dir1/dir2/dir3/original-file.tif"
        with create_and_delete_temp_file(filename) as local_path:
            actual_upload_path = upload_file(
                bucket_name=bucket, local_path=Path(local_path), src_uri=src_uri
            )
            self.assertEqual(
                actual_upload_path, f"gs://{bucket}/dir1/dir2/dir3/{filename}"
            )

    @patch("docker.torchserve_handler.storage")
    def test_upload_file_gcloud_calls(self, mock_storage):
        bucket = "fake-bucket"
        filename = "fake-file.tif"
        src_uri = "gs://original-bucket/original-file.tif"
        with create_and_delete_temp_file(filename) as local_path:
            upload_file(
                bucket_name=bucket, local_path=Path(local_path), src_uri=src_uri
            )

        mock_storage.Client.assert_called()
        mock_storage_client = mock_storage.Client()
        mock_storage_client.bucket.assert_called_once()
        mock_storage_bucket = mock_storage_client.bucket(bucket)
        mock_storage_bucket.blob.assert_called_once()
        mock_storage_bucket.blob(filename).upload_from_filename.assert_called_once()

from unittest.mock import MagicMock, patch

import pytest
from requests import RequestException

from gliomoda.constants import WEIGHTS_FOLDER
from gliomoda.weights import (
    _download_weights,
    _extract_archive,
    _get_zenodo_metadata_and_archive_url,
    check_weights_path,
)


@pytest.fixture
def mock_zenodo_metadata():
    return {"version": "1.0.0"}, "https://fakeurl.com/archive.zip"


@pytest.fixture
def mock_weights_folder(tmp_path):
    weights_path = tmp_path / "weights"
    weights_path.mkdir()
    return weights_path


@patch("gliomoda.weights.requests.get")
def test_get_zenodo_metadata_and_archive_url(mock_get, mock_zenodo_metadata):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "metadata": mock_zenodo_metadata[0],
        "links": {"archive": mock_zenodo_metadata[1]},
    }
    mock_get.return_value = mock_response

    metadata, archive_url = _get_zenodo_metadata_and_archive_url()
    assert metadata["version"] == "1.0.0"
    assert archive_url == "https://fakeurl.com/archive.zip"


@patch("gliomoda.weights.requests.get")
def test_get_zenodo_metadata_and_archive_url_failure(mock_get):
    mock_get.side_effect = RequestException()
    assert _get_zenodo_metadata_and_archive_url() == None


@patch("gliomoda.weights._get_latest_version_folder_name", return_value=None)
@patch("gliomoda.weights._get_zenodo_metadata_and_archive_url", return_value=None)
@patch("gliomoda.weights.logger.error")
def test_check_weights_path_no_local_no_meta(
    mock_sys_exit, mock_get_meta, mock_get_latest_version
):
    with pytest.raises(SystemExit):
        check_weights_path()
    mock_sys_exit.assert_called_once_with(
        "Weights not found locally and Zenodo could not be reached. Exiting..."
    )


@patch("gliomoda.weights._get_latest_version_folder_name", return_value=None)
@patch("gliomoda.weights._get_zenodo_metadata_and_archive_url")
@patch("gliomoda.weights._download_weights")
def test_check_weights_path_no_local(
    mock_download, mock_zenodo_meta, mock_weights_folder
):
    mock_zenodo_meta.return_value = ({"version": "1.0.0"}, "https://fakeurl.com")
    mock_download.return_value = mock_weights_folder / "weights_v1.0.0"

    weights_path = check_weights_path()
    assert weights_path == mock_weights_folder / "weights_v1.0.0"


@patch(
    "gliomoda.weights._get_latest_version_folder_name", return_value="weights_v1.0.0"
)
@patch("gliomoda.weights.logger.info")
@patch("gliomoda.weights._get_zenodo_metadata_and_archive_url")
def test_check_weights_path_latest_local(
    mock_zenodo_meta, mock_logger_info, mock_weights_folder
):
    mock_zenodo_meta.return_value = ({"version": "1.0.0"}, "https://fakeurl.com")

    weights_path = check_weights_path()
    assert weights_path == WEIGHTS_FOLDER / "weights_v1.0.0"
    mock_logger_info.assert_called_with(f"Latest weights (1.0.0) are already present.")


@patch(
    "gliomoda.weights._get_latest_version_folder_name", return_value="weights_v1.0.0"
)
@patch("gliomoda.weights.logger.info", return_value=None)
@patch("gliomoda.weights._get_zenodo_metadata_and_archive_url")
@patch("gliomoda.weights._download_weights")
def test_check_weights_path_old_local(
    mock_download, mock_zenodo_meta, mock_logger_info, mock_weights_folder
):
    mock_zenodo_meta.return_value = ({"version": "2.0.0"}, "https://fakeurl.com")
    mock_download.return_value = mock_weights_folder / "weights_v2.0.0"

    weights_path = check_weights_path()

    mock_logger_info.assert_called_with(
        "New weights available on Zenodo (2.0.0). Deleting old and fetching new weights..."
    )

    assert weights_path == mock_weights_folder / "weights_v2.0.0"


@patch("gliomoda.weights._extract_archive")
@patch("gliomoda.weights.requests.get")
def test_download_weights(mock_get, mock_extract_archive, mock_zenodo_metadata):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content = lambda chunk_size: [b"data"]
    mock_get.return_value = mock_response

    weights_path = _download_weights(mock_zenodo_metadata[0], mock_zenodo_metadata[1])
    assert weights_path.exists()


@patch("gliomoda.weights.zipfile.ZipFile")
def test_extract_archive(mock_zipfile, tmp_path):
    mock_response = MagicMock()
    mock_response.iter_content = lambda chunk_size: [b"data"]
    record_folder = tmp_path / "weights_v1.0.0"
    record_folder.mkdir()

    dummy_zip = record_folder / "archive.zip"
    dummy_zip.touch()

    mock_zip = MagicMock()
    mock_zip.namelist.return_value = ["file1.txt", "file2.txt"]
    mock_zip.__enter__.return_value = mock_zip
    mock_zipfile.return_value = mock_zip

    _extract_archive(mock_response, record_folder)

    mock_zip.extract.assert_called()

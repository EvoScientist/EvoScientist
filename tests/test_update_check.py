"""Tests for EvoScientist.update_check module."""

import json
import time
from unittest.mock import MagicMock, patch

from EvoScientist.update_check import (
    CACHE_TTL,
    _parse_version,
    get_latest_version,
    is_update_available,
)


class TestParseVersion:
    """Tests for _parse_version."""

    def test_basic(self):
        assert _parse_version("1.2.3") == (1, 2, 3)

    def test_single_digit(self):
        assert _parse_version("5") == (5,)

    def test_whitespace(self):
        assert _parse_version("  0.0.2  ") == (0, 0, 2)

    def test_comparison(self):
        assert _parse_version("0.0.3") > _parse_version("0.0.2")
        assert _parse_version("0.1.0") > _parse_version("0.0.9")
        assert _parse_version("1.0.0") > _parse_version("0.9.9")

    def test_equal(self):
        assert _parse_version("0.0.2") == _parse_version("0.0.2")


class TestGetLatestVersion:
    """Tests for get_latest_version."""

    def test_fresh_cache_hit(self, tmp_path):
        cache_file = tmp_path / "latest_version.json"
        cache_file.write_text(
            json.dumps({"version": "1.0.0", "checked_at": time.time()})
        )
        with patch("EvoScientist.update_check.CACHE_FILE", cache_file):
            assert get_latest_version() == "1.0.0"

    def test_stale_cache_fetches_pypi(self, tmp_path):
        cache_file = tmp_path / "latest_version.json"
        cache_file.write_text(
            json.dumps(
                {
                    "version": "0.0.1",
                    "checked_at": time.time() - CACHE_TTL - 1,
                }
            )
        )
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"info": {"version": "2.0.0"}}
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch("EvoScientist.update_check.CACHE_FILE", cache_file),
            patch("EvoScientist.update_check.CACHE_DIR", tmp_path),
            patch("urllib.request.urlopen", return_value=mock_resp),
        ):
            assert get_latest_version() == "2.0.0"

    def test_no_cache_fetches_pypi(self, tmp_path):
        cache_file = tmp_path / "latest_version.json"
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"info": {"version": "3.0.0"}}
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch("EvoScientist.update_check.CACHE_FILE", cache_file),
            patch("EvoScientist.update_check.CACHE_DIR", tmp_path),
            patch("urllib.request.urlopen", return_value=mock_resp),
        ):
            assert get_latest_version() == "3.0.0"
            # Cache should have been written
            assert cache_file.exists()
            data = json.loads(cache_file.read_text())
            assert data["version"] == "3.0.0"

    def test_network_error_returns_none(self, tmp_path):
        cache_file = tmp_path / "latest_version.json"
        with (
            patch("EvoScientist.update_check.CACHE_FILE", cache_file),
            patch("urllib.request.urlopen", side_effect=OSError("network down")),
        ):
            assert get_latest_version() is None

    def test_corrupt_cache_recovers(self, tmp_path):
        cache_file = tmp_path / "latest_version.json"
        cache_file.write_text("not valid json!!!")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"info": {"version": "1.5.0"}}
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch("EvoScientist.update_check.CACHE_FILE", cache_file),
            patch("EvoScientist.update_check.CACHE_DIR", tmp_path),
            patch("urllib.request.urlopen", return_value=mock_resp),
        ):
            assert get_latest_version() == "1.5.0"


class TestIsUpdateAvailable:
    """Tests for is_update_available."""

    def test_newer_version_available(self):
        with (
            patch("EvoScientist.update_check.get_latest_version", return_value="9.9.9"),
            patch("EvoScientist.update_check._installed_version", return_value="0.0.2"),
        ):
            available, latest = is_update_available()
            assert available is True
            assert latest == "9.9.9"

    def test_same_version(self):
        with (
            patch("EvoScientist.update_check.get_latest_version", return_value="0.0.2"),
            patch("EvoScientist.update_check._installed_version", return_value="0.0.2"),
        ):
            available, _latest = is_update_available()
            assert available is False

    def test_older_pypi_version(self):
        with (
            patch("EvoScientist.update_check.get_latest_version", return_value="0.0.1"),
            patch("EvoScientist.update_check._installed_version", return_value="0.0.2"),
        ):
            available, _latest = is_update_available()
            assert available is False

    def test_pypi_unreachable(self):
        with patch("EvoScientist.update_check.get_latest_version", return_value=None):
            available, latest = is_update_available()
            assert available is False
            assert latest is None

    def test_invalid_version_string(self):
        with (
            patch("EvoScientist.update_check.get_latest_version", return_value="abc"),
            patch("EvoScientist.update_check._installed_version", return_value="0.0.2"),
        ):
            available, _latest = is_update_available()
            assert available is False

"""Tests for EvoScientist.update_check module."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from EvoScientist.update_check import (
    CACHE_TTL,
    CN_SIMPLE,
    fetch_versions,
    get_latest_version,
    is_update_available,
    newest_release,
)


def _response(payload):
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestNewestRelease:
    def test_numeric_order_not_string_order(self):
        assert newest_release(["0.3.9", "0.3.10", "0.3.2"]) == "0.3.10"

    def test_skips_prereleases_and_garbage(self):
        assert newest_release(["0.3.6", "0.4.0rc1", "0.4.0.dev1", "x"]) == "0.3.6"

    def test_none_without_releases(self):
        assert newest_release(["1.0a1"]) is None


class TestFetchVersions:
    def test_reads_the_json_simple_api(self):
        resp = _response({"meta": {}, "versions": ["0.3.5", "0.3.6"]})
        with patch("urllib.request.urlopen", return_value=resp) as urlopen:
            assert fetch_versions(CN_SIMPLE, 5) == ["0.3.5", "0.3.6"]
        request = urlopen.call_args.args[0]
        assert request.full_url == f"{CN_SIMPLE}/evoscientist/"
        assert request.get_header("Accept") == "application/vnd.pypi.simple.v1+json"


class TestGetLatestVersion:
    """Tests for get_latest_version."""

    @pytest.fixture(autouse=True)
    def _default_mirror(self, monkeypatch):
        class Config:
            mirror = "default"

        monkeypatch.setattr("EvoScientist.config.load_config", lambda: Config())

    def test_refresh_skips_fresh_cache(self, tmp_path):
        cache_file = tmp_path / "latest_version.json"
        cache_file.write_text(
            json.dumps({"version": "1.0.0", "checked_at": time.time()})
        )
        with (
            patch("EvoScientist.update_check.CACHE_FILE", cache_file),
            patch("EvoScientist.update_check.CACHE_DIR", tmp_path),
            patch(
                "urllib.request.urlopen",
                return_value=_response({"info": {"version": "2.0.0"}}),
            ),
        ):
            assert get_latest_version(refresh=True) == "2.0.0"

    def test_cn_mirror_reads_tsinghua(self, tmp_path):
        cache_file = tmp_path / "latest_version.json"
        resp = _response({"versions": ["0.3.10", "0.3.9", "0.4.0rc1"]})
        with (
            patch("EvoScientist.update_check.CACHE_FILE", cache_file),
            patch("EvoScientist.update_check.CACHE_DIR", tmp_path),
            patch("urllib.request.urlopen", return_value=resp) as urlopen,
        ):
            assert get_latest_version(mirror="cn") == "0.3.10"
        assert urlopen.call_args.args[0].full_url.startswith(CN_SIMPLE)

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

    def test_prerelease_installed_sees_final_release(self):
        with (
            patch("EvoScientist.update_check.get_latest_version", return_value="0.4.0"),
            patch(
                "EvoScientist.update_check._installed_version", return_value="0.4.0rc1"
            ),
        ):
            assert is_update_available() == (True, "0.4.0")

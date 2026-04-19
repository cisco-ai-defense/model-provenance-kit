# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Stress tests for provenancekit.services.download — offline (mocked network)."""

import hashlib
import io
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from provenancekit.config.settings import Settings
from provenancekit.services.download import (
    _find_by_family,
    _progress_bar,
    _safe_rename,
    _sha256_file,
    download_deep_signals,
    has_deep_signals,
    show_deep_signals_status,
)

# ── Helpers ────────────────────────────────────────────────────────


def _build_zip(
    files: dict[str, bytes],
    *,
    include_macos_junk: bool = False,
    include_unsafe_path: bool = False,
) -> bytes:
    """Build an in-memory zip archive with the given file entries."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
        if include_macos_junk:
            zf.writestr("__MACOSX/._by-family/phi/junk", b"junk")
            zf.writestr("by-family/phi/._hidden", b"hidden")
        if include_unsafe_path:
            zf.writestr("../../../etc/passwd", b"root:x:0:0")
            zf.writestr("/absolute/path.parquet", b"evil")
    buf.seek(0)
    return buf.read()


def _good_zip() -> bytes:
    """A minimal valid zip with by-family structure."""
    return _build_zip(
        {
            "by-family/phi/phi-1_5_deep-signals.parquet": b"fake-parquet-phi",
            "by-family/llama/llama-7b_deep-signals.parquet": b"fake-parquet-llama",
        }
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mock_opener(zip_bytes: bytes) -> MagicMock:
    """Create a mock for urllib.request.build_opener that returns zip_bytes."""
    resp = MagicMock()
    resp.headers = {"Content-Length": str(len(zip_bytes))}
    stream = io.BytesIO(zip_bytes)
    resp.read = stream.read
    resp.close = MagicMock()
    opener = MagicMock()
    opener.open = MagicMock(return_value=resp)
    builder = MagicMock(return_value=opener)
    return builder


def _setup_db_root(tmp_path: Path) -> Path:
    """Create the minimum db_root directory tree."""
    db_root = tmp_path / "database"
    ds_dir = db_root / "features" / "deep-signals"
    ds_dir.mkdir(parents=True)
    return db_root


# ── Unit tests: helper functions ──────────────────────────────────


class TestProgressBar:
    def test_known_total(self) -> None:
        result = _progress_bar(50 * 1024 * 1024, 100 * 1024 * 1024)
        assert "50%" in result
        assert "50.0" in result
        assert "100.0" in result

    def test_zero_total_no_crash(self) -> None:
        result = _progress_bar(1024 * 1024, 0)
        assert "1.0 MB downloaded" in result

    def test_negative_total(self) -> None:
        result = _progress_bar(0, -1)
        assert "downloaded" in result

    def test_complete(self) -> None:
        result = _progress_bar(100, 100)
        assert "100%" in result


class TestSha256File:
    def test_correct_hash(self, tmp_path: Path) -> None:
        data = b"hello world"
        p = tmp_path / "test.bin"
        p.write_bytes(data)
        assert _sha256_file(p) == hashlib.sha256(data).hexdigest()

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        assert _sha256_file(p) == hashlib.sha256(b"").hexdigest()


class TestSafeRename:
    def test_basic_rename(self, tmp_path: Path) -> None:
        src = tmp_path / "src_dir"
        src.mkdir()
        (src / "file.txt").write_text("data")
        dst = tmp_path / "dst_dir"
        _safe_rename(src, dst)
        assert dst.is_dir()
        assert (dst / "file.txt").read_text() == "data"
        assert not src.exists()

    def test_rename_file(self, tmp_path: Path) -> None:
        src = tmp_path / "a.txt"
        src.write_text("hello")
        dst = tmp_path / "b.txt"
        _safe_rename(src, dst)
        assert dst.read_text() == "hello"


class TestFindByFamily:
    def test_direct_child(self, tmp_path: Path) -> None:
        bf = tmp_path / "by-family"
        bf.mkdir()
        assert _find_by_family(tmp_path) == bf

    def test_nested_in_wrapper(self, tmp_path: Path) -> None:
        bf = tmp_path / "deep-signals" / "by-family"
        bf.mkdir(parents=True)
        assert _find_by_family(tmp_path) == bf

    def test_missing(self, tmp_path: Path) -> None:
        (tmp_path / "some_other_dir").mkdir()
        assert _find_by_family(tmp_path) is None

    def test_file_not_dir(self, tmp_path: Path) -> None:
        (tmp_path / "by-family").write_text("not a dir")
        assert _find_by_family(tmp_path) is None


class TestHasDeepSignals:
    def test_with_parquets(self, tmp_path: Path) -> None:
        db_root = tmp_path / "db"
        bf = db_root / "features" / "deep-signals" / "by-family" / "phi"
        bf.mkdir(parents=True)
        (bf / "test.parquet").write_bytes(b"data")
        assert has_deep_signals(db_root) is True

    def test_empty_by_family(self, tmp_path: Path) -> None:
        db_root = tmp_path / "db"
        bf = db_root / "features" / "deep-signals" / "by-family"
        bf.mkdir(parents=True)
        assert has_deep_signals(db_root) is False

    def test_missing_dir(self, tmp_path: Path) -> None:
        db_root = tmp_path / "db"
        db_root.mkdir()
        assert has_deep_signals(db_root) is False

    def test_non_parquet_files_only(self, tmp_path: Path) -> None:
        db_root = tmp_path / "db"
        bf = db_root / "features" / "deep-signals" / "by-family" / "phi"
        bf.mkdir(parents=True)
        (bf / "readme.txt").write_text("not a parquet")
        assert has_deep_signals(db_root) is False


# ── Integration tests: download_deep_signals ──────────────────────


class TestDownloadDeepSignals:
    """Tests that mock the network but exercise real filesystem logic."""

    def _run_download(
        self,
        tmp_path: Path,
        zip_bytes: bytes | None = None,
        *,
        update: bool = False,
        verify: bool = True,
        sha_override: str | None = None,
    ) -> tuple[int, Path]:
        """Helper: run download_deep_signals with mocked network."""
        if zip_bytes is None:
            zip_bytes = _good_zip()

        db_root = _setup_db_root(tmp_path)
        mock_builder = _mock_opener(zip_bytes)

        sha_patch = sha_override or _sha256(zip_bytes)
        test_settings = Settings(
            db_root=db_root,
            hf_deep_signals_sha256=sha_patch,
        )

        with patch("urllib.request.build_opener", mock_builder):
            rc = download_deep_signals(
                db_root,
                update=update,
                verify=verify,
                settings=test_settings,
            )
        return rc, db_root

    def test_fresh_install_success(self, tmp_path: Path) -> None:
        rc, db_root = self._run_download(tmp_path)
        assert rc == 0

        by_family = db_root / "features" / "deep-signals" / "by-family"
        assert by_family.is_dir()
        parquets = list(by_family.rglob("*.parquet"))
        assert len(parquets) == 2

        marker = db_root / "features" / "deep-signals" / ".deep-signals-installed"
        assert marker.exists()
        text = marker.read_text()
        assert "installed_from=" in text
        assert "sha256=" in text
        assert "installed_at=" in text

    def test_skip_when_already_installed(self, tmp_path: Path) -> None:
        rc, db_root = self._run_download(tmp_path)
        assert rc == 0

        rc2 = download_deep_signals(db_root, update=False)
        assert rc2 == 0

    def test_update_replaces_existing(self, tmp_path: Path) -> None:
        rc, db_root = self._run_download(tmp_path)
        assert rc == 0

        new_zip = _build_zip(
            {
                "by-family/bert/bert-base_deep-signals.parquet": b"new-bert-data",
            }
        )
        mock_builder = _mock_opener(new_zip)
        sha = _sha256(new_zip)
        test_settings = Settings(
            db_root=db_root,
            hf_deep_signals_sha256=sha,
        )

        with patch("urllib.request.build_opener", mock_builder):
            rc2 = download_deep_signals(
                db_root,
                update=True,
                verify=True,
                settings=test_settings,
            )
        assert rc2 == 0

        by_family = db_root / "features" / "deep-signals" / "by-family"
        parquets = list(by_family.rglob("*.parquet"))
        assert len(parquets) == 1
        assert parquets[0].name == "bert-base_deep-signals.parquet"

        assert not (by_family.parent / "by-family.bak").exists()

    def test_integrity_check_failure(self, tmp_path: Path) -> None:
        rc, _ = self._run_download(
            tmp_path,
            sha_override="0000000000000000000000000000000000000000000000000000000000000000",
        )
        assert rc == 1

    def test_skip_verify(self, tmp_path: Path) -> None:
        rc, db_root = self._run_download(
            tmp_path,
            sha_override="wrong_hash_should_not_matter",
            verify=False,
        )
        assert rc == 0
        assert has_deep_signals(db_root)

    def test_bad_zip_file(self, tmp_path: Path) -> None:
        rc, _ = self._run_download(tmp_path, zip_bytes=b"not a zip file", verify=False)
        assert rc == 1

    def test_zip_without_by_family(self, tmp_path: Path) -> None:
        bad_zip = _build_zip({"some_random_file.txt": b"hello"})
        rc, _ = self._run_download(tmp_path, zip_bytes=bad_zip, verify=False)
        assert rc == 1

    def test_macos_junk_filtered(self, tmp_path: Path) -> None:
        zip_bytes = _build_zip(
            {"by-family/phi/test.parquet": b"data"},
            include_macos_junk=True,
        )
        rc, db_root = self._run_download(tmp_path, zip_bytes=zip_bytes, verify=False)
        assert rc == 0

        by_family = db_root / "features" / "deep-signals" / "by-family"
        all_files = [f.name for f in by_family.rglob("*") if f.is_file()]
        assert "junk" not in all_files
        assert "._hidden" not in all_files
        assert "test.parquet" in all_files

    def test_unsafe_paths_skipped(self, tmp_path: Path) -> None:
        zip_bytes = _build_zip(
            {"by-family/phi/test.parquet": b"data"},
            include_unsafe_path=True,
        )
        rc, db_root = self._run_download(tmp_path, zip_bytes=zip_bytes, verify=False)
        assert rc == 0

        assert not (tmp_path / "etc").exists()
        assert not Path("/absolute/path.parquet").exists()

    def test_network_failure(self, tmp_path: Path) -> None:
        from urllib.error import URLError

        db_root = _setup_db_root(tmp_path)
        opener = MagicMock()
        opener.open = MagicMock(side_effect=URLError("Connection refused"))
        with patch(
            "urllib.request.build_opener",
            return_value=opener,
        ):
            rc = download_deep_signals(db_root)
        assert rc == 1

    def test_http_404(self, tmp_path: Path) -> None:
        from urllib.error import HTTPError

        db_root = _setup_db_root(tmp_path)
        opener = MagicMock()
        opener.open = MagicMock(
            side_effect=HTTPError(
                url="https://example.com",
                code=404,
                msg="Not Found",
                hdrs=MagicMock(),  # type: ignore[arg-type]
                fp=None,
            ),
        )
        with patch(
            "urllib.request.build_opener",
            return_value=opener,
        ):
            rc = download_deep_signals(db_root)
        assert rc == 1

    def test_tmp_dir_cleaned_on_success(self, tmp_path: Path) -> None:
        rc, db_root = self._run_download(tmp_path)
        assert rc == 0

        ds_dir = db_root / "features" / "deep-signals"
        tmp_dirs = [
            d for d in ds_dir.iterdir() if d.is_dir() and d.name.startswith("tmp")
        ]
        assert len(tmp_dirs) == 0

    def test_tmp_dir_cleaned_on_failure(self, tmp_path: Path) -> None:
        rc, db_root = self._run_download(tmp_path, zip_bytes=b"corrupt", verify=False)
        assert rc == 1

        ds_dir = db_root / "features" / "deep-signals"
        tmp_dirs = [
            d for d in ds_dir.iterdir() if d.is_dir() and d.name.startswith("tmp")
        ]
        assert len(tmp_dirs) == 0

    def test_empty_zip(self, tmp_path: Path) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w"):
            pass
        buf.seek(0)
        rc, _ = self._run_download(tmp_path, zip_bytes=buf.read(), verify=False)
        assert rc == 1

    def test_db_root_created_if_missing(self, tmp_path: Path) -> None:
        db_root = tmp_path / "brand_new" / "database"
        zip_bytes = _good_zip()
        mock_builder = _mock_opener(zip_bytes)
        test_settings = Settings(
            db_root=db_root,
            hf_deep_signals_sha256=_sha256(zip_bytes),
        )

        with patch("urllib.request.build_opener", mock_builder):
            rc = download_deep_signals(
                db_root,
                verify=True,
                settings=test_settings,
            )
        assert rc == 0
        assert has_deep_signals(db_root)

    def test_wrapped_zip_structure(self, tmp_path: Path) -> None:
        """Zip has a top-level wrapper dir like deep-signals/by-family/..."""
        zip_bytes = _build_zip(
            {
                "deep-signals/by-family/phi/test.parquet": b"data",
            }
        )
        rc, db_root = self._run_download(tmp_path, zip_bytes=zip_bytes, verify=False)
        assert rc == 0
        assert has_deep_signals(db_root)


# ── Integration tests: show_deep_signals_status ───────────────────


class TestShowStatus:
    def test_not_installed(self, tmp_path: Path, capsys: Any) -> None:
        db_root = tmp_path / "db"
        db_root.mkdir()
        rc = show_deep_signals_status(db_root)
        assert rc == 0
        out = capsys.readouterr().out
        assert "NOT installed" in out

    def test_installed(self, tmp_path: Path, capsys: Any) -> None:
        db_root = tmp_path / "db"
        ds = db_root / "features" / "deep-signals"
        bf = ds / "by-family" / "phi"
        bf.mkdir(parents=True)
        (bf / "test.parquet").write_bytes(b"data")
        marker = ds / ".deep-signals-installed"
        marker.write_text(
            "installed_from=https://example.com\n"
            "sha256=abc\n"
            "installed_at=2026-01-01T00:00:00Z\n",
        )

        rc = show_deep_signals_status(db_root)
        assert rc == 0
        out = capsys.readouterr().out
        assert "installed at:" in out.lower() or "installed_from" in out
        assert "Parquet files: 1" in out
        assert "phi" in out

    def test_installed_empty_by_family(self, tmp_path: Path, capsys: Any) -> None:
        db_root = tmp_path / "db"
        ds = db_root / "features" / "deep-signals"
        bf = ds / "by-family"
        bf.mkdir(parents=True)
        marker = ds / ".deep-signals-installed"
        marker.write_text("installed_from=test\n")

        rc = show_deep_signals_status(db_root)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Parquet files: 0" in out

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

"""Download deep-signal weight fingerprints from Hugging Face.

Downloads a zip archive containing parquet-based weight fingerprints
and extracts them into ``features/deep-signals/by-family/`` inside the
provenance database.  The ``features/base/`` bundles and ``catalog/``
are never touched -- they ship with the repo.
"""

import hashlib
import shutil
import ssl
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request

import structlog

from provenancekit.config.settings import Settings

log = structlog.get_logger()


def _get_settings() -> Settings:
    """Return a Settings instance, created lazily at call time."""
    return Settings()


def _ssl_context() -> ssl.SSLContext:
    """Create an SSL context that verifies certificates."""
    ctx = ssl.create_default_context()
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Block redirects to non-HTTPS URLs to prevent SSRF."""

    def redirect_request(  # noqa: PLR0913
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        if not newurl.startswith("https://"):
            raise URLError(f"Blocked redirect to non-HTTPS URL: {newurl}")
        return super().redirect_request(
            req,
            fp,
            code,
            msg,
            headers,
            newurl,
        )


_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2  # seconds; doubled each attempt
_MAX_EXTRACT_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB
_MAX_EXTRACT_FILES = 50_000


def _progress_bar(current: int, total: int, width: int = 40) -> str:
    """Render a single-line progress indicator."""
    mb_cur = current / (1024 * 1024)
    if total <= 0:
        return f"  {mb_cur:.1f} MB downloaded"
    pct = current / total
    filled = int(width * pct)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    mb_tot = total / (1024 * 1024)
    return f"  [{bar}] {pct:.0%}  {mb_cur:.1f}/{mb_tot:.1f} MB"


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def _safe_rename(src: Path, dst: Path) -> None:
    """Rename *src* to *dst*, falling back to shutil.move on Windows."""
    try:
        src.rename(dst)
    except OSError:
        shutil.move(str(src), str(dst))


def has_deep_signals(db_root: Path) -> bool:
    """Return True if at least one parquet exists under deep-signals."""
    by_family = db_root / "features" / "deep-signals" / "by-family"
    if not by_family.is_dir():
        return False
    return any(by_family.rglob("*.parquet"))


def download_deep_signals(
    db_root: Path,
    *,
    update: bool = False,
    verify: bool = True,
    settings: Settings | None = None,
) -> int:
    """Download deep-signals.zip from HF and install into *db_root*.

    Returns 0 on success, 1 on failure.
    """
    if settings is None:
        settings = _get_settings()
    t0 = time.perf_counter()

    deep_signals_dir = db_root / "features" / "deep-signals"
    by_family_dir = deep_signals_dir / "by-family"
    marker_file = deep_signals_dir / ".deep-signals-installed"

    if marker_file.exists() and not update:
        log.info("deep_signals_already_installed", path=str(deep_signals_dir))
        print(f"Deep-signal fingerprints already installed at: {deep_signals_dir}")
        print("Use --update to re-download and replace with the latest.")
        return 0

    deep_signals_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir_obj = tempfile.mkdtemp(dir=str(deep_signals_dir))
    tmp_dir = Path(tmp_dir_obj)
    try:
        return _download_and_install(
            deep_signals_dir=deep_signals_dir,
            by_family_dir=by_family_dir,
            marker_file=marker_file,
            tmp_dir=tmp_dir,
            verify=verify,
            t0=t0,
            settings=settings,
        )
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _download_and_install(
    *,
    deep_signals_dir: Path,
    by_family_dir: Path,
    marker_file: Path,
    tmp_dir: Path,
    verify: bool,
    t0: float,
    settings: Settings,
) -> int:
    """Core download, verify, extract, and atomic-swap logic."""
    zip_path = tmp_dir / "deep-signals.zip"
    hf_url = settings.hf_deep_signals_url
    hf_sha256 = settings.hf_deep_signals_sha256

    if not hf_url.startswith("https://"):
        log.error("insecure_download_url", url=hf_url)
        print(f"  Error: download URL must use HTTPS (got: {hf_url})", file=sys.stderr)
        return 1

    # ── Download ──────────────────────────────────────────────
    print("Downloading deep-signal fingerprints from Hugging Face ...")
    print(f"  URL: {hf_url}")
    log.info("download_start", url=hf_url)

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            req = Request(
                hf_url,
                headers={"User-Agent": "provenancekit/0.1"},
            )
            opener = urllib.request.build_opener(
                _HttpsOnlyRedirectHandler,
                urllib.request.HTTPSHandler(context=_ssl_context()),
            )
            resp = opener.open(req, timeout=300)
        except (HTTPError, URLError, OSError) as exc:
            last_exc = exc  # noqa: F841
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF_BASE**attempt
                log.warning(
                    "download_retry",
                    attempt=attempt,
                    max_retries=_MAX_RETRIES,
                    wait_s=wait,
                    error=str(exc),
                )
                print(
                    f"  Attempt {attempt}/{_MAX_RETRIES} failed: {exc}. "
                    f"Retrying in {wait}s ...",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            log.error("download_failed", error=str(exc), attempts=_MAX_RETRIES)
            print(
                f"  Download failed after {_MAX_RETRIES} attempts: {exc}",
                file=sys.stderr,
            )
            return 1

        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        t_dl = time.perf_counter()

        try:
            with zip_path.open("wb") as f:
                while True:
                    chunk = resp.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    print(
                        f"\r{_progress_bar(downloaded, total_size)}",
                        end="",
                        flush=True,
                        file=sys.stderr,
                    )
        except OSError as exc:
            resp.close()
            last_exc = exc  # noqa: F841
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF_BASE**attempt
                log.warning(
                    "download_read_retry",
                    attempt=attempt,
                    max_retries=_MAX_RETRIES,
                    wait_s=wait,
                    error=str(exc),
                )
                print(
                    f"\n  Read error on attempt {attempt}/{_MAX_RETRIES}: {exc}. "
                    f"Retrying in {wait}s ...",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            log.error("download_read_failed", error=str(exc), attempts=_MAX_RETRIES)
            print(
                f"\n  Download read failed after {_MAX_RETRIES} attempts: {exc}",
                file=sys.stderr,
            )
            return 1
        finally:
            resp.close()

        break  # success

    print(file=sys.stderr)
    log.info(
        "download_complete",
        size_mb=round(downloaded / (1024 * 1024), 1),
        elapsed=_elapsed(t_dl),
    )
    print(
        f"  Download complete: {downloaded / (1024 * 1024):.1f} MB ({_elapsed(t_dl)})"
    )

    # ── Verify integrity ──────────────────────────────────────
    if verify:
        print("  Verifying integrity (SHA-256) ...")
        actual_hash = _sha256_file(zip_path)
        if actual_hash != hf_sha256:
            log.error(
                "integrity_check_failed",
                expected=hf_sha256,
                actual=actual_hash,
            )
            print(
                f"  Integrity check FAILED!\n"
                f"    expected: {hf_sha256}\n"
                f"    actual:   {actual_hash}",
                file=sys.stderr,
            )
            return 1
        log.info("integrity_ok", sha256=actual_hash)
        print(f"  Integrity OK: {actual_hash[:16]}...")

    # ── Extract ───────────────────────────────────────────────
    print(f"  Extracting to: {deep_signals_dir}")
    t_ext = time.perf_counter()

    extract_dir = tmp_dir / "extracted"
    extract_dir.mkdir()

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                log.error("corrupt_zip_entry", entry=bad)
                print(f"  Corrupt file in zip: {bad}", file=sys.stderr)
                return 1

            names = [
                n
                for n in zf.namelist()
                if not n.startswith("__MACOSX")
                and not n.split("/")[-1].startswith("._")
            ]

            if len(names) > _MAX_EXTRACT_FILES:
                log.error("too_many_zip_entries", count=len(names))
                print(
                    f"  Error: zip has {len(names)} entries "
                    f"(limit: {_MAX_EXTRACT_FILES})",
                    file=sys.stderr,
                )
                return 1

            total_uncompressed = sum(
                zf.getinfo(n).file_size for n in names if not zf.getinfo(n).is_dir()
            )
            if total_uncompressed > _MAX_EXTRACT_BYTES:
                size_gb = total_uncompressed / (1024**3)
                log.error("zip_too_large", bytes=total_uncompressed)
                print(
                    f"  Error: uncompressed size {size_gb:.1f} GB exceeds limit",
                    file=sys.stderr,
                )
                return 1

            log.info("extracting", entries=len(names))
            for i, name in enumerate(names, 1):
                info = zf.getinfo(name)
                member_path = Path(name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    log.warning("skipping_unsafe_path", path=name)
                    continue
                if info.external_attr >> 28 == 0xA:
                    log.warning("skipping_symlink", path=name)
                    continue
                zf.extract(name, extract_dir)
                if i % 50 == 0 or i == len(names):
                    print(
                        f"\r  Extracted {i}/{len(names)} files",
                        end="",
                        flush=True,
                        file=sys.stderr,
                    )

            print(file=sys.stderr)
    except zipfile.BadZipFile as exc:
        log.error("bad_zip_file", error=str(exc))
        print(f"  Invalid zip file: {exc}", file=sys.stderr)
        return 1

    log.info("extraction_complete", elapsed=_elapsed(t_ext))
    print(f"  Extraction complete ({_elapsed(t_ext)})")

    # ── Locate the by-family tree inside the extracted archive ─
    extracted_by_family = _find_by_family(extract_dir)
    if extracted_by_family is None:
        log.error("by_family_not_found_in_archive")
        print(
            "  Error: archive does not contain a by-family/ directory.",
            file=sys.stderr,
        )
        return 1

    # ── Atomic swap ───────────────────────────────────────────
    backup_dir = deep_signals_dir / "by-family.bak"

    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    if by_family_dir.exists():
        _safe_rename(by_family_dir, backup_dir)

    try:
        _safe_rename(extracted_by_family, by_family_dir)
    except Exception as exc:
        log.warning("catalog_rename_failed_rolling_back", error=str(exc))
        if backup_dir.exists() and not by_family_dir.exists():
            _safe_rename(backup_dir, by_family_dir)
        raise

    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)

    # ── Write marker ──────────────────────────────────────────
    marker_file.write_text(
        f"installed_from={hf_url}\n"
        f"sha256={hf_sha256}\n"
        f"installed_at={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n",
        encoding="utf-8",
    )

    # ── Summary ───────────────────────────────────────────────
    total_files = sum(1 for _ in by_family_dir.rglob("*.parquet"))
    total_size_mb = sum(
        f.stat().st_size for f in by_family_dir.rglob("*") if f.is_file()
    ) / (1024 * 1024)

    print()
    print("=" * 60)
    print("  Deep-signal fingerprints installed successfully!")
    print(f"  Location:  {deep_signals_dir}")
    print(f"  Parquets:  {total_files}")
    print(f"  Size:      {total_size_mb:.1f} MB")
    print(f"  Time:      {_elapsed(t0)}")
    print("=" * 60)

    log.info(
        "install_complete",
        parquets=total_files,
        size_mb=round(total_size_mb, 1),
        elapsed=_elapsed(t0),
    )
    return 0


def _find_by_family(root: Path) -> Path | None:
    """Walk *root* to locate the ``by-family`` directory.

    The zip may contain a top-level wrapper directory, so we search
    rather than hard-code the exact nesting.  Only matches directories
    named exactly ``by-family`` whose name equals the search term
    (avoids matching e.g. ``by-family-old/``).
    """
    for candidate in root.rglob("by-family"):
        if candidate.is_dir() and candidate.name == "by-family":
            return candidate
    return None


def show_deep_signals_status(db_root: Path) -> int:
    """Print the current deep-signals installation status.

    Returns 0 always (informational).
    """
    deep_signals_dir = db_root / "features" / "deep-signals"
    marker = deep_signals_dir / ".deep-signals-installed"

    if not marker.exists():
        print(f"Deep-signal fingerprints NOT installed at: {deep_signals_dir}")
        print(
            "Run: provenancekit download-deepsignals-fingerprint",
        )
        return 0

    print(f"Deep-signal fingerprints installed at: {deep_signals_dir}")
    print(marker.read_text(encoding="utf-8").strip())

    by_family = deep_signals_dir / "by-family"
    parquet_count = sum(1 for _ in by_family.rglob("*.parquet"))
    total_size = sum(f.stat().st_size for f in by_family.rglob("*") if f.is_file()) / (
        1024 * 1024
    )

    print(f"Parquet files: {parquet_count}")
    print(f"Total size:    {total_size:.1f} MB")

    families: set[str] = set()
    if by_family.is_dir():
        families = {d.name for d in by_family.iterdir() if d.is_dir()}
    if families:
        print(f"Families:      {', '.join(sorted(families))}")

    return 0

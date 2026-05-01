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

"""Disk + in-memory caching for extracted model features.

Provides a ``CacheService`` class that wraps:

* **In-memory dict** — fast repeated lookups within a session.
* **Disk JSON files** — persistence across sessions.

The public API uses typed ``CachedEntry`` objects.  Internally, disk
files are plain JSON via Pydantic ``model_dump`` / ``model_validate``.
"""

import hashlib
import hmac
import json
import os
import secrets
import tempfile
import threading
from pathlib import Path

import structlog

from provenancekit.config.settings import Settings
from provenancekit.exceptions import CacheError
from provenancekit.models.results import CachedEntry

log = structlog.get_logger()


class NullCache:
    """No-op cache that never stores or returns anything.

    Used when ``--no-cache`` is passed on the CLI so that the rest of
    the pipeline can call ``.get()`` / ``.put()`` without guards.
    """

    def get(self, model_id: str) -> CachedEntry | None:  # noqa: ARG002
        """Always return ``None`` (cache disabled)."""
        return None

    def put(self, model_id: str, entry: CachedEntry) -> None:  # noqa: ARG002
        """Do nothing (cache disabled)."""

    def clear(self, model_id: str | None = None) -> None:  # noqa: ARG002
        """Do nothing (cache disabled)."""


class CacheService:
    """Thread-safe, two-layer (memory + disk) feature cache."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Initialise the cache.

        Args:
            cache_dir: Directory for JSON cache files.  Defaults to
                ``Settings().cache_dir`` (``~/.provenancekit/cache``).
        """
        self._dir = cache_dir if cache_dir is not None else Settings().cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, CachedEntry] = {}
        self._lock = threading.Lock()

    def _hmac_key(self) -> bytes:
        """Load or create a per-installation HMAC key."""
        key_path = self._dir / ".cache_key"
        try:
            return key_path.read_bytes()
        except FileNotFoundError:
            key = secrets.token_bytes(32)
            key_path.write_bytes(key)
            key_path.chmod(0o600)
            return key

    def _compute_hmac(self, data: bytes) -> str:
        """Return hex HMAC-SHA256 of *data*."""
        return hmac.new(self._hmac_key(), data, hashlib.sha256).hexdigest()

    # ── public API ────────────────────────────────────────────────

    def get(self, model_id: str) -> CachedEntry | None:
        """Return a cached entry, checking memory first then disk.

        Cache failures are non-fatal: a :class:`CacheError` from the
        disk layer is caught, logged, and ``None`` is returned.
        """
        model_id = model_id.strip()
        with self._lock:
            if model_id in self._mem:
                return self._mem[model_id]

        try:
            entry = self._load_disk(model_id)
        except CacheError:
            return None
        if entry is not None:
            with self._lock:
                # Re-check after disk load to avoid overwriting a concurrent put().
                if model_id not in self._mem:
                    self._mem[model_id] = entry
                else:
                    entry = self._mem[model_id]
        return entry

    def put(self, model_id: str, entry: CachedEntry) -> None:
        """Store an entry in memory and persist to disk.

        If the model already has a disk entry with ``vocab`` data and
        the new entry does not include ``vocab``, the existing vocab
        is preserved (merge-on-write).

        Cache write failures are non-fatal: a :class:`CacheError` from
        the disk layer is caught and logged at warning level.
        """
        model_id = model_id.strip()
        entry = self._merge_vocab(model_id, entry)

        with self._lock:
            self._mem[model_id] = entry
        try:
            self._save_disk(model_id, entry)
        except CacheError as exc:
            log.warning("cache_write_failed", model_id=model_id, error=str(exc))

    def clear(self, model_id: str | None = None) -> None:
        """Remove entries from the in-memory cache.

        Args:
            model_id: Specific model to evict.  ``None`` clears all.
        """
        with self._lock:
            if model_id is not None:
                self._mem.pop(model_id.strip(), None)
            else:
                self._mem.clear()

    # ── internal helpers ──────────────────────────────────────────

    def _cache_path(self, model_id: str) -> Path:
        """Return the JSON file path for *model_id*."""
        safe = model_id.strip().replace("/", "__").replace("\\", "__")
        result = (self._dir / f"{safe}.json").resolve()
        if not result.is_relative_to(self._dir.resolve()):
            raise CacheError(
                f"Invalid model_id would escape cache directory: {model_id}",
                details={"model_id": model_id},
            )
        return result

    def _load_disk(self, model_id: str) -> CachedEntry | None:
        """Read a cached entry from disk and verify its HMAC.

        Entries without a valid ``_hmac`` field are treated as missing
        (returns ``None``) so the pipeline re-extracts gracefully.

        Raises:
            CacheError: On corrupt or unreadable cache files.
        """
        path = self._cache_path(model_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            stored_hmac = raw.pop("_hmac", None)
            if stored_hmac is None:
                log.warning("cache_hmac_missing", path=str(path))
                return None
            payload = json.dumps(raw, default=str, sort_keys=True).encode("utf-8")
            if not hmac.compare_digest(stored_hmac, self._compute_hmac(payload)):
                log.warning("cache_hmac_mismatch", path=str(path))
                return None
            return CachedEntry.model_validate(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("corrupt_cache_file", path=str(path))
            raise CacheError(
                f"Corrupt cache file: {path}",
                details={"path": str(path), "model_id": model_id},
            ) from exc

    def _save_disk(self, model_id: str, entry: CachedEntry) -> None:
        """Write a cached entry to disk atomically.

        Writes to a temporary file in the same directory, then uses
        ``os.replace`` to atomically swap in the new content.  This
        prevents truncated / corrupt cache files if the process is
        killed mid-write.

        Raises:
            CacheError: On filesystem write failures.
        """
        path = self._cache_path(model_id)
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._dir),
                suffix=".tmp",
            )
            try:
                payload = json.dumps(
                    entry.model_dump(),
                    default=str,
                    sort_keys=True,
                ).encode("utf-8")
                sig = self._compute_hmac(payload)
                signed = json.loads(payload)
                signed["_hmac"] = sig
                with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                    json.dump(signed, tmp_f, default=str)
                Path(tmp_path).replace(path)
            except BaseException:
                Path(tmp_path).unlink()
                raise
        except OSError as exc:
            log.warning("cache_write_failed", path=str(path))
            raise CacheError(
                f"Failed to write cache file: {path}",
                details={"path": str(path), "model_id": model_id},
            ) from exc

    def _merge_vocab(self, model_id: str, entry: CachedEntry) -> CachedEntry:
        """Preserve existing vocab when the new entry omits it."""
        if entry.vocab is not None:
            return entry
        with self._lock:
            mem_entry = self._mem.get(model_id)
        if mem_entry is not None and mem_entry.vocab is not None:
            return entry.model_copy(update={"vocab": mem_entry.vocab})
        try:
            existing = self._load_disk(model_id)
        except CacheError as exc:
            log.warning("cache_vocab_merge_skipped", model_id=model_id, error=str(exc))
            return entry
        if existing is not None and existing.vocab is not None:
            return entry.model_copy(update={"vocab": existing.vocab})
        return entry

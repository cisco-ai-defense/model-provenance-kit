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

"""Tests for provenancekit.services.cache — all offline (filesystem only)."""

import json
from pathlib import Path

from provenancekit.models.results import CachedEntry
from provenancekit.services.cache import CacheService


def _sample_entry(model_id: str = "org/test-model") -> CachedEntry:
    return CachedEntry(
        model_id=model_id,
        mfi={"model_type": "llama", "arch_hash": "abc123"},
        tfv={"vocab_size": 32000, "tokenizer_class": "LlamaTokenizer"},
        vocab=["!", "a", "the"],
    )


# ── CachedEntry model ────────────────────────────────────────────


class TestCachedEntry:
    def test_defaults(self) -> None:
        entry = CachedEntry(model_id="gpt2")
        assert entry.mfi is None
        assert entry.tfv is None
        assert entry.vocab is None

    def test_with_data(self) -> None:
        entry = _sample_entry()
        assert entry.mfi is not None
        assert entry.mfi["model_type"] == "llama"
        assert entry.vocab == ["!", "a", "the"]

    def test_roundtrip_json(self) -> None:
        entry = _sample_entry()
        raw = entry.model_dump()
        restored = CachedEntry.model_validate(raw)
        assert restored == entry


# ── CacheService disk operations ──────────────────────────────────


class TestCacheDisk:
    def test_put_and_get_roundtrip(self, tmp_path: Path) -> None:
        svc = CacheService(cache_dir=tmp_path)
        entry = _sample_entry()
        svc.put("org/test-model", entry)
        loaded = svc.get("org/test-model")
        assert loaded is not None
        assert loaded.model_id == "org/test-model"
        assert loaded.mfi == entry.mfi
        assert loaded.tfv == entry.tfv
        assert loaded.vocab == entry.vocab

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        svc = CacheService(cache_dir=tmp_path)
        assert svc.get("nonexistent/model") is None

    def test_filename_sanitizes_slashes(self, tmp_path: Path) -> None:
        svc = CacheService(cache_dir=tmp_path)
        path = svc._cache_path("org/model-name")
        assert "/" not in path.name
        assert path.name == "org__model-name.json"

    def test_put_preserves_existing_vocab(self, tmp_path: Path) -> None:
        svc = CacheService(cache_dir=tmp_path)
        full = _sample_entry()
        svc.put("org/test-model", full)

        no_vocab = CachedEntry(
            model_id="org/test-model",
            mfi={"model_type": "llama", "arch_hash": "def456"},
            tfv={"vocab_size": 32000},
        )
        svc.put("org/test-model", no_vocab)

        loaded = svc.get("org/test-model")
        assert loaded is not None
        assert loaded.mfi == {"model_type": "llama", "arch_hash": "def456"}
        assert loaded.vocab == ["!", "a", "the"]

    def test_disk_file_is_valid_json(self, tmp_path: Path) -> None:
        svc = CacheService(cache_dir=tmp_path)
        svc.put("gpt2", CachedEntry(model_id="gpt2", mfi={"model_type": "gpt2"}))
        path = svc._cache_path("gpt2")
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["model_id"] == "gpt2"
        assert data["mfi"]["model_type"] == "gpt2"

    def test_corrupt_cache_returns_none(self, tmp_path: Path) -> None:
        svc = CacheService(cache_dir=tmp_path)
        path = svc._cache_path("bad-model")
        path.write_text("NOT VALID JSON {{{", encoding="utf-8")
        assert svc.get("bad-model") is None

    def test_creates_cache_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "cache"
        svc = CacheService(cache_dir=nested)
        assert nested.is_dir()
        svc.put("m", CachedEntry(model_id="m"))
        assert svc.get("m") is not None


# ── CacheService in-memory cache ──────────────────────────────────


class TestCacheMemory:
    def test_mem_cache_hit(self, tmp_path: Path) -> None:
        svc = CacheService(cache_dir=tmp_path)
        entry = _sample_entry()
        svc.put("org/test-model", entry)

        disk_path = svc._cache_path("org/test-model")
        disk_path.unlink()

        loaded = svc.get("org/test-model")
        assert loaded is not None
        assert loaded.mfi == entry.mfi

    def test_clear_single_model(self, tmp_path: Path) -> None:
        svc = CacheService(cache_dir=tmp_path)
        svc.put("a", CachedEntry(model_id="a", mfi={"k": "v"}))
        svc.put("b", CachedEntry(model_id="b", mfi={"k": "v"}))

        svc.clear("a")

        assert svc.get("b") is not None
        loaded_a = svc.get("a")
        assert loaded_a is not None
        assert loaded_a.mfi == {"k": "v"}

    def test_clear_all(self, tmp_path: Path) -> None:
        svc = CacheService(cache_dir=tmp_path)
        svc.put("a", CachedEntry(model_id="a"))
        svc.put("b", CachedEntry(model_id="b"))

        svc.clear()

        assert svc.get("a") is not None
        assert svc.get("b") is not None

    def test_clear_all_removes_mem_only(self, tmp_path: Path) -> None:
        svc = CacheService(cache_dir=tmp_path)
        svc.put("x", CachedEntry(model_id="x"))

        svc.clear()
        svc._cache_path("x").unlink()

        assert svc.get("x") is None

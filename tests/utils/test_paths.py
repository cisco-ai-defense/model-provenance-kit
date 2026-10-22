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

"""Tests for provenancekit.utils.paths."""

from pathlib import Path

import pytest

from provenancekit.utils.paths import UnsafeShardPathError, safe_shard_join


def _hf_cache_snapshot(tmp_path: Path) -> Path:
    """Mimic the HF cache: snapshot files are symlinks into ../../blobs/."""
    blobs = tmp_path / "models--org--m" / "blobs"
    snap = tmp_path / "models--org--m" / "snapshots" / "abc123"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    (blobs / "deadbeef").write_bytes(b"x")
    (snap / "model-00001-of-00002.safetensors").symlink_to("../../blobs/deadbeef")
    return snap


class TestSafeShardJoin:
    def test_plain_name(self, tmp_path: Path) -> None:
        out = safe_shard_join(str(tmp_path), "model-00001-of-00002.safetensors")
        assert out == str(tmp_path / "model-00001-of-00002.safetensors")

    def test_subdirectory_allowed(self, tmp_path: Path) -> None:
        out = safe_shard_join(str(tmp_path), "shards/a.safetensors")
        assert out == str(tmp_path / "shards" / "a.safetensors")

    def test_hf_cache_symlink_allowed_and_not_resolved(self, tmp_path: Path) -> None:
        snap = _hf_cache_snapshot(tmp_path)
        out = safe_shard_join(str(snap), "model-00001-of-00002.safetensors")
        assert out == str(snap / "model-00001-of-00002.safetensors")
        assert Path(out).read_bytes() == b"x"

    @pytest.mark.parametrize(
        "shard",
        [
            "",
            "   ",
            "a\x00b",
            "/etc/passwd",
            "\\windows\\system32",
            "C:\\evil.bin",
            "c:evil.bin",
            "../escape.bin",
            "shards/../../escape.bin",
            "..\\escape.bin",
            "shards\\..\\..\\escape.bin",
            "..",
        ],
    )
    def test_rejects_unsafe_names(self, tmp_path: Path, shard: str) -> None:
        with pytest.raises(UnsafeShardPathError):
            safe_shard_join(str(tmp_path), shard)

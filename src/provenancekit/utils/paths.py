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

"""Path helpers for shard files named by untrusted index JSON."""

import os
import re
from pathlib import PurePosixPath, PureWindowsPath

_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class UnsafeShardPathError(ValueError):
    """A shard name from an index file would escape the model directory."""


def safe_shard_join(base_dir: str, shard: str) -> str:
    r"""Join *shard* onto *base_dir*, rejecting names that could escape it.

    The check is lexical and the returned path is not resolved: Hugging
    Face cache snapshots store shards as symlinks into ``../../blobs/``,
    so resolving would reject every cached sharded model.

    Raises:
        UnsafeShardPathError: For empty or absolute names, drive letters,
            NUL bytes, or ``..`` components (``/`` or ``\`` separated).
    """
    if not isinstance(shard, str) or not shard.strip() or "\x00" in shard:
        raise UnsafeShardPathError(f"Invalid shard name: {shard!r}")
    if (
        PurePosixPath(shard).is_absolute()
        or PureWindowsPath(shard).is_absolute()
        or shard.startswith(("/", "\\"))
        or _DRIVE_RE.match(shard)
    ):
        raise UnsafeShardPathError(f"Absolute shard path not allowed: {shard!r}")
    if ".." in re.split(r"[\\/]", shard):
        raise UnsafeShardPathError(f"Shard path escapes model directory: {shard!r}")

    base = os.path.normpath(os.path.abspath(base_dir))  # noqa: PTH100
    joined = os.path.normpath(os.path.join(base, shard))  # noqa: PTH118
    if os.path.commonpath([base, joined]) != base:
        raise UnsafeShardPathError(f"Shard path escapes model directory: {shard!r}")
    return joined

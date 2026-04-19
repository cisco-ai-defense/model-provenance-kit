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

"""ProvenanceKit — Model provenance detection toolkit."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from provenancekit.config.settings import Settings
    from provenancekit.core.scanner import ModelProvenanceScanner
    from provenancekit.exceptions import (
        CacheError,
        ExtractionError,
        ModelLoadError,
        ProvenanceError,
    )

__all__ = [
    "CacheError",
    "ExtractionError",
    "ModelLoadError",
    "ModelProvenanceScanner",
    "ProvenanceError",
    "Settings",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ModelProvenanceScanner": ("provenancekit.core.scanner", "ModelProvenanceScanner"),
    "Settings": ("provenancekit.config.settings", "Settings"),
    "CacheError": ("provenancekit.exceptions", "CacheError"),
    "ExtractionError": ("provenancekit.exceptions", "ExtractionError"),
    "ModelLoadError": ("provenancekit.exceptions", "ModelLoadError"),
    "ProvenanceError": ("provenancekit.exceptions", "ProvenanceError"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

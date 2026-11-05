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

"""Tests for resuming ``scripts/build_database.py``."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from provenancekit.config.constants import EXTRACTOR_VERSION
from provenancekit.config.settings import Settings
from provenancekit.models.storage import FeatureBundle


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "build_database.py"
    spec = importlib.util.spec_from_file_location("build_database", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_database = _load_script()


class TestIsBuilt:
    @pytest.mark.parametrize(
        ("version", "params", "transformers_version", "built"),
        [
            (EXTRACTOR_VERSION, "current", "5.5.0", True),
            (EXTRACTOR_VERSION, "current", "5.5.3", True),
            ("1", "current", "5.5.0", False),
            (EXTRACTOR_VERSION, {"anchor_k": 1, "wvc_subsample": 1}, "5.5.0", False),
            (EXTRACTOR_VERSION, None, "5.5.0", False),
            (EXTRACTOR_VERSION, "current", "5.6.0", False),
            (EXTRACTOR_VERSION, "current", None, False),
        ],
    )
    def test_bundle_must_match_extractor_and_settings(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        version: str,
        params: dict[str, int] | str | None,
        transformers_version: str | None,
        built: bool,
    ) -> None:
        monkeypatch.setattr(build_database.transformers, "__version__", "5.5.0")
        settings = Settings()
        if params == "current":
            params = settings.extraction_params()
        job = SimpleNamespace(bundle_rel=lambda: "features/a.json")
        bundle = FeatureBundle(
            extractor_version=version,
            extraction_params=params,
            transformers_version=transformers_version,
        )
        (tmp_path / "features").mkdir()
        (tmp_path / "features/a.json").write_text(bundle.model_dump_json())
        assert build_database._is_built(tmp_path, job, settings) is built

    def test_missing_bundle_not_built(self, tmp_path: Path) -> None:
        job = SimpleNamespace(bundle_rel=lambda: "features/a.json")
        assert build_database._is_built(tmp_path, job, Settings()) is False

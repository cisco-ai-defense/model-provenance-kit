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

"""Pydantic-settings based application settings."""

from pathlib import Path

from pydantic_settings import BaseSettings

_PACKAGE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Runtime-configurable knobs for ProvenanceKit.

    Every field can be overridden via an environment variable
    prefixed with ``PROVENANCEKIT_``.
    """

    cache_dir: Path = Path.home() / ".provenancekit" / "cache"
    db_root: Path = _PACKAGE_DIR / "data" / "database"
    scan_top_k: int = 3
    scan_threshold: float = 0.50
    huge_model_params: float = 10e9
    anchor_k: int = 64
    wvc_subsample: int = 4096
    trust_remote_code: bool = False
    dev_mode: bool = False

    hf_dataset_repo: str = "cisco-ai/model-provenance-kit"
    hf_deep_signals_url: str = ""
    hf_deep_signals_sha256: str = (
        "b94040b668bea03e44b95a69c8554046d68fe0325b54ce64141bbf40765e48e9"
    )

    def model_post_init(self, __context: object) -> None:
        """Derive the download URL from the repo name when not set explicitly."""
        if not self.hf_deep_signals_url:
            self.hf_deep_signals_url = (
                f"https://huggingface.co/datasets/{self.hf_dataset_repo}"
                "/resolve/main/deep-signals.zip"
            )

    model_config = {"env_prefix": "PROVENANCEKIT_"}

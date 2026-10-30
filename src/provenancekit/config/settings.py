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

from provenancekit.config.constants import EXTRACTOR_VERSION

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
        "8021d7e631fa6a1c04707ce29d9bfca6b75d5bddd44ec768919a18430ed39bb7"
    )

    def model_post_init(self, __context: object) -> None:
        """Derive the download URL from the repo name when not set explicitly."""
        if not self.hf_deep_signals_url:
            # Versioned file name: earlier releases pin the SHA-256 of
            # ``deep-signals.zip``, so that file must stay unchanged.
            self.hf_deep_signals_url = (
                f"https://huggingface.co/datasets/{self.hf_dataset_repo}"
                f"/resolve/main/deep-signals-v{EXTRACTOR_VERSION}.zip"
            )

    def extraction_params(self) -> dict[str, int]:
        """Settings that change extracted feature values."""
        return {"anchor_k": self.anchor_k, "wvc_subsample": self.wvc_subsample}

    model_config = {"env_prefix": "PROVENANCEKIT_"}

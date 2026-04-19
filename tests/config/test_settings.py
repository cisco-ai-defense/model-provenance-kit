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

from pathlib import Path

from provenancekit.config.settings import Settings


def test_defaults():
    s = Settings()
    assert s.anchor_k == 64
    assert s.wvc_subsample == 4096
    assert s.huge_model_params == 10e9
    assert isinstance(s.cache_dir, Path)
    assert s.hf_dataset_repo
    assert s.hf_dataset_repo in s.hf_deep_signals_url
    assert "deep-signals.zip" in s.hf_deep_signals_url
    assert len(s.hf_deep_signals_sha256) == 64


def test_env_override(monkeypatch):
    monkeypatch.setenv("PROVENANCEKIT_ANCHOR_K", "32")
    s = Settings()
    assert s.anchor_k == 32


def test_env_override_cache_dir(monkeypatch):
    monkeypatch.setenv("PROVENANCEKIT_CACHE_DIR", "/tmp/pk_test")
    s = Settings()
    assert s.cache_dir == Path("/tmp/pk_test")


def test_hf_url_derived_from_repo(monkeypatch):
    monkeypatch.setenv("PROVENANCEKIT_HF_DATASET_REPO", "my-org/my-repo")
    s = Settings()
    assert s.hf_dataset_repo == "my-org/my-repo"
    assert "my-org/my-repo" in s.hf_deep_signals_url


def test_hf_url_explicit_override(monkeypatch):
    monkeypatch.setenv(
        "PROVENANCEKIT_HF_DEEP_SIGNALS_URL", "https://example.com/data.zip"
    )
    s = Settings()
    assert s.hf_deep_signals_url == "https://example.com/data.zip"

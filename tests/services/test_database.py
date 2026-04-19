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

"""Tests for provenancekit.services.database — offline (filesystem only)."""

import json
from pathlib import Path

import numpy as np
import pytest

from provenancekit.services.database import DatabaseError, DatabaseService


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _build_mini_db(root: Path) -> None:
    """Create a minimal valid DB with one family and one asset."""
    _write_json(
        root / "catalog" / "manifest.json",
        {
            "updated_at": "2026-01-01T00:00:00Z",
            "shard_strategy": "by_family",
            "shards": [
                {
                    "family_id": "bloom",
                    "shard_path": "catalog/by-family/bloom.json",
                    "asset_count": 1,
                },
            ],
        },
    )
    _write_json(
        root / "catalog" / "by-family" / "bloom.json",
        {
            "shard_id": "test-shard-001",
            "updated_at": "2026-01-01T00:00:00Z",
            "family": {
                "family_id": "bloom",
                "display_name": "BLOOM",
                "publisher": "BigScience",
                "baseline_model_ids": ["bloom-560m"],
            },
            "assets": [
                {
                    "asset_id": "bloom-560m__hf-safetensors",
                    "model_id": "bloom-560m",
                    "canonical_name": "bigscience/bloom-560m",
                    "format": "safetensors",
                    "feature_path": "features/base/bloom_features.json",
                    "param_bucket": "<=1B",
                },
            ],
        },
    )
    _write_json(
        root / "features" / "base" / "bloom_features.json",
        {
            "bundle_id": "b001",
            "model_id": "bloom-560m",
            "asset_id": "bloom-560m__hf-safetensors",
            "family_id": "bloom",
            "status": "ok",
            "mfi": {
                "model_type": "bloom",
                "arch_hash": "aaa",
                "family_hash": "abc123",
                "hidden_size": 1024,
                "num_hidden_layers": 24,
            },
            "tfv": {"vocab_size": 250680, "tokenizer_class": "TokenizersBackend"},
            "signals": {
                "hidden_size": 1024,
                "num_layers": 24,
                "eas_anchor_count": 64,
                "nlf_num_layers": 100,
            },
            "artifact_refs": [
                {
                    "artifact_id": "bloom-deep",
                    "type": "deep_signals",
                    "path": "features/deep/bloom_deep.parquet",
                    "signals": ["eas_self_sim"],
                },
            ],
        },
    )


class TestCatalogLoading:
    def test_load_catalog_counts(self, tmp_path: Path) -> None:
        db_root = tmp_path / "db"
        _build_mini_db(db_root)
        svc = DatabaseService(db_root)
        cat = svc.load_catalog()
        assert len(cat.families) == 1
        assert len(cat.assets) == 1

    def test_family_id_populated_on_assets(self, tmp_path: Path) -> None:
        db_root = tmp_path / "db"
        _build_mini_db(db_root)
        svc = DatabaseService(db_root)
        cat = svc.load_catalog()
        asset = cat.assets["bloom-560m__hf-safetensors"]
        assert asset.family_id == "bloom"

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        svc = DatabaseService(tmp_path / "empty")
        with pytest.raises(DatabaseError, match="manifest not found"):
            svc.load_catalog()

    def test_catalog_cached_on_second_call(self, tmp_path: Path) -> None:
        db_root = tmp_path / "db"
        _build_mini_db(db_root)
        svc = DatabaseService(db_root)
        cat1 = svc.load_catalog()
        cat2 = svc.load_catalog()
        assert cat1 is cat2


class TestFeatureBundle:
    def test_load_feature_bundle(self, tmp_path: Path) -> None:
        db_root = tmp_path / "db"
        _build_mini_db(db_root)
        svc = DatabaseService(db_root)
        bundle = svc.load_feature_bundle("features/base/bloom_features.json")
        assert bundle.model_id == "bloom-560m"
        assert bundle.mfi["arch_hash"] == "aaa"
        assert bundle.signals.hidden_size == 1024

    def test_missing_bundle_raises(self, tmp_path: Path) -> None:
        svc = DatabaseService(tmp_path)
        with pytest.raises(DatabaseError, match="not found"):
            svc.load_feature_bundle("nonexistent.json")


class TestDeepSignals:
    def _write_parquet(self, path: Path) -> None:
        """Write a tiny parquet with eas_self_sim (2x2) and a 1D signal."""
        import pandas as pd

        rows = [
            {"signal": "eas_self_sim", "layer": None, "row": 0, "col": 0, "value": 1.0},
            {"signal": "eas_self_sim", "layer": None, "row": 0, "col": 1, "value": 0.5},
            {"signal": "eas_self_sim", "layer": None, "row": 1, "col": 0, "value": 0.5},
            {"signal": "eas_self_sim", "layer": None, "row": 1, "col": 1, "value": 1.0},
            {
                "signal": "lep_profile",
                "layer": None,
                "row": 0,
                "col": None,
                "value": 0.8,
            },
            {
                "signal": "lep_profile",
                "layer": None,
                "row": 1,
                "col": None,
                "value": 0.6,
            },
            {
                "signal": "wvc_layer_sigs",
                "layer": 0,
                "row": 0,
                "col": None,
                "value": 0.9,
            },
            {
                "signal": "wvc_layer_sigs",
                "layer": 0,
                "row": 1,
                "col": None,
                "value": 0.7,
            },
            {
                "signal": "wvc_layer_sigs",
                "layer": 1,
                "row": 0,
                "col": None,
                "value": 0.3,
            },
        ]
        df = pd.DataFrame(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    def test_eas_reconstructed_as_2d(self, tmp_path: Path) -> None:
        pq = tmp_path / "signals.parquet"
        self._write_parquet(pq)
        svc = DatabaseService(tmp_path)
        signals = svc.load_deep_signals("signals.parquet")
        assert "eas_self_sim" in signals
        eas = signals["eas_self_sim"]
        assert isinstance(eas, np.ndarray)
        assert eas.shape == (2, 2)
        assert eas[0, 1] == pytest.approx(0.5)

    def test_wvc_reconstructed_as_layer_dict(self, tmp_path: Path) -> None:
        pq = tmp_path / "signals.parquet"
        self._write_parquet(pq)
        svc = DatabaseService(tmp_path)
        signals = svc.load_deep_signals("signals.parquet")
        wvc = signals["wvc_layer_sigs"]
        assert isinstance(wvc, dict)
        assert len(wvc) == 2
        assert 0 in wvc and 1 in wvc
        assert wvc[0].shape == (2,)

    def test_1d_signal_sorted_by_row(self, tmp_path: Path) -> None:
        pq = tmp_path / "signals.parquet"
        self._write_parquet(pq)
        svc = DatabaseService(tmp_path)
        signals = svc.load_deep_signals("signals.parquet")
        lep = signals["lep_profile"]
        assert lep.shape == (2,)
        assert lep[0] == pytest.approx(0.8)

    def test_missing_parquet_returns_empty(self, tmp_path: Path) -> None:
        svc = DatabaseService(tmp_path)
        assert svc.load_deep_signals("nope.parquet") == {}

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

"""File-first provenance database access layer.

Reads the sharded catalog, feature bundles, and deep-signals parquets
stored under ``provenancekit/data/database/``.  All heavy numpy
reconstruction from parquet is handled here so callers receive typed
Pydantic models or numpy arrays directly.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from provenancekit.exceptions import ProvenanceError
from provenancekit.models.signals import WeightSignalFeatures
from provenancekit.models.storage import (
    AssetRecord,
    CatalogManifest,
    CatalogShard,
    FamilyRecord,
    FeatureBundle,
)

log = structlog.get_logger()


class DatabaseError(ProvenanceError):
    """Raised when the provenance database cannot be read or is corrupt."""


class Catalog:
    """In-memory representation of the full catalog after loading.

    Provides O(1) lookups by family_id and asset_id.
    """

    def __init__(  # noqa: D107
        self,
        families: dict[str, FamilyRecord],
        assets: dict[str, AssetRecord],
    ) -> None:
        self.families = families
        self.assets = assets


class DatabaseService:
    """Read-only access to the file-first provenance database.

    Args:
        db_root: Path to the ``provenancekit/data/database/`` directory containing
            ``catalog/manifest.json`` and the ``features/`` tree.
    """

    def __init__(self, db_root: Path) -> None:  # noqa: D107
        self._root = db_root
        self._catalog: Catalog | None = None
        self._family_names: dict[str, str] = {}

    # ── catalog ───────────────────────────────────────────────────

    def load_catalog(self) -> Catalog:
        """Load the full catalog from the sharded manifest.

        Populates ``family_id`` on asset records at read time (it is
        not stored redundantly in the JSON).

        Raises:
            DatabaseError: When the manifest or a shard file cannot be read.
        """
        if self._catalog is not None:
            return self._catalog

        t0 = time.perf_counter()
        manifest_path = self._root / "catalog" / "manifest.json"
        if not manifest_path.exists():
            raise DatabaseError(
                f"Catalog manifest not found: {manifest_path}",
                details={"path": str(manifest_path)},
            )

        manifest = CatalogManifest.model_validate(
            self._read_json(manifest_path),
        )

        families: dict[str, FamilyRecord] = {}
        assets: dict[str, AssetRecord] = {}

        for shard_ref in manifest.shards:
            shard_path = self._root / shard_ref.shard_path
            if not shard_path.exists():
                log.warning(
                    "shard_missing",
                    family_id=shard_ref.family_id,
                    path=str(shard_path),
                )
                continue

            shard = CatalogShard.model_validate(self._read_json(shard_path))
            fid = shard.family.family_id
            families[fid] = shard.family
            self._family_names[fid] = shard.family.display_name

            for a in shard.assets:
                a.family_id = fid
                assets[a.asset_id] = a

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        log.info(
            "catalog_loaded",
            families=len(families),
            assets=len(assets),
            elapsed_ms=elapsed,
        )

        self._catalog = Catalog(
            families=families,
            assets=assets,
        )
        return self._catalog

    # ── feature bundles ───────────────────────────────────────────

    def load_feature_bundle(self, feature_path: str) -> FeatureBundle:
        """Load a single ``features.json`` file.

        Args:
            feature_path: Relative path within the DB root
                (e.g. ``features/base/by-family/bloom/…_features.json``).

        Raises:
            DatabaseError: When the file is missing or invalid.
        """
        full_path = (self._root / feature_path).resolve()
        if not full_path.is_relative_to(self._root.resolve()):
            raise DatabaseError(
                f"Feature path escapes database root: {feature_path}",
                details={"path": str(full_path), "db_root": str(self._root)},
            )
        if not full_path.exists():
            raise DatabaseError(
                f"Feature bundle not found: {full_path}",
                details={"path": str(full_path)},
            )
        return FeatureBundle.model_validate(self._read_json(full_path))

    # ── deep-signals parquet ──────────────────────────────────────

    def load_deep_signals(
        self,
        parquet_path: str,
    ) -> dict[str, np.ndarray | dict[int, np.ndarray]]:
        """Load a deep-signals parquet and reconstruct numpy arrays.

        The parquet uses a long-form schema with columns
        ``signal, layer, row, col, value``.  This method pivots each
        signal back into its native shape:

        * ``eas_self_sim`` → 2D float32 matrix (k x k)
        * ``wvc_layer_sigs`` → dict mapping layer index to 1D float32
        * all others → 1D float32 vector sorted by ``row``

        Args:
            parquet_path: Relative path within the DB root.

        Returns:
            Mapping of signal name to reconstructed numpy array(s).
            Empty dict if the parquet does not exist.
        """
        full_path = (self._root / parquet_path).resolve()
        if not full_path.is_relative_to(self._root.resolve()):
            log.warning("parquet_path_escapes_root", path=parquet_path)
            return {}
        if not full_path.exists():
            return {}

        df = pd.read_parquet(full_path)
        signals: dict[str, np.ndarray | dict[int, np.ndarray]] = {}

        for signal_name, subset in df.groupby("signal", sort=False):
            if signal_name == "eas_self_sim":
                k = int(subset["row"].max()) + 1
                mat = np.zeros((k, k), dtype=np.float32)
                rows = subset["row"].values
                cols = subset["col"].fillna(0).astype(int).values
                mat[rows, cols] = subset["value"].values  # type: ignore[index,assignment]
                signals[str(signal_name)] = mat

            elif signal_name == "wvc_layer_sigs":
                layer_dict: dict[int, np.ndarray] = {}
                for layer_idx in sorted(subset["layer"].dropna().unique()):
                    layer_sub = subset[subset["layer"] == layer_idx]
                    layer_dict[int(layer_idx)] = layer_sub.sort_values("row")[
                        "value"
                    ].values.astype(np.float32)
                signals[str(signal_name)] = layer_dict

            else:
                signals[str(signal_name)] = subset.sort_values("row")[  # type: ignore[assignment]
                    "value"
                ].values.astype(np.float32)

        return signals

    def reconstruct_weight_features(
        self,
        deep_signals: dict[str, np.ndarray | dict[int, np.ndarray]],
        bundle: FeatureBundle,
    ) -> WeightSignalFeatures:
        """Build a ``WeightSignalFeatures`` from DB parquet data.

        Combines the reconstructed numpy arrays from
        :meth:`load_deep_signals` with scalar metadata from the
        feature bundle's ``signals`` summary.
        """
        sigs = bundle.signals
        return WeightSignalFeatures(
            hidden_size=sigs.hidden_size,
            num_layers=sigs.num_layers,
            eas_self_sim=deep_signals.get("eas_self_sim"),  # type: ignore[arg-type]
            eas_anchor_count=sigs.eas_anchor_count,
            nlf_vector=deep_signals.get("nlf_vector"),  # type: ignore[arg-type]
            nlf_mode=sigs.nlf_mode,
            nlf_num_layers=sigs.nlf_num_layers,
            lep_profile=deep_signals.get("lep_profile"),  # type: ignore[arg-type]
            end_histogram=deep_signals.get("end_histogram"),  # type: ignore[arg-type]
            wsp_signature=deep_signals.get("wsp_signature"),  # type: ignore[arg-type]
            wvc_layer_sigs=deep_signals.get("wvc_layer_sigs"),  # type: ignore[arg-type]
        )

    # ── lookup helpers ────────────────────────────────────────────

    def get_family_display_name(self, family_id: str) -> str:
        """Return the human-readable family name, falling back to the id."""
        if family_id in self._family_names:
            return self._family_names[family_id]

        shard_path = self._root / "catalog" / "by-family" / f"{family_id}.json"
        if not shard_path.exists():
            return family_id
        try:
            shard = CatalogShard.model_validate(self._read_json(shard_path))
            name = shard.family.display_name
            self._family_names[family_id] = name
            return name
        except Exception as exc:  # noqa: BLE001
            log.debug("family_name_lookup_failed", family_id=family_id, error=str(exc))
            return family_id

    # ── internal ──────────────────────────────────────────────────

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        """Read and parse a JSON file."""
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

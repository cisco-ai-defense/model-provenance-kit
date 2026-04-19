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

"""Pydantic models for the file-first provenance database.

Covers catalog entities (families, assets), feature bundles, artifact
references, and the sharded manifest structure that maps to the on-disk
``provenancekit/data/database/`` layout.

Schema reference::

    provenancekit/data/database/
    ├── catalog/
    │   ├── manifest.json            → CatalogManifest
    │   └── by-family/<fid>.json     → CatalogShard
    └── features/
        ├── base/by-family/…         → FeatureBundle
        └── deep-signals/by-family/… → (parquet, not modelled here)
"""

from typing import Any

from pydantic import BaseModel, Field

# ── Catalog entities ─────────────────────────────────────────────


class FamilyRecord(BaseModel):
    """A model family in the catalog (e.g. BLOOM, Gemma, Qwen).

    ``baseline_model_ids`` lists the canonical base models against
    which derivatives are compared.
    """

    family_id: str
    display_name: str
    publisher: str
    family_hash: str = ""
    baseline_model_ids: list[str] = Field(default_factory=list)


class AssetRecord(BaseModel):
    """A downloadable asset (weights file) associated with a model.

    ``param_bucket`` enables Stage 1 structural filtering without
    loading the full feature bundle.  ``family_id`` is populated at
    load time from the enclosing shard's family record.
    """

    asset_id: str
    model_id: str
    canonical_name: str = ""
    format: str = "other"
    source_uri: str = ""
    checksums: dict[str, str] = Field(default_factory=dict)
    feature_path: str = ""
    param_bucket: str = ""
    family_id: str = ""


# ── Catalog shard and manifest ───────────────────────────────────


class CatalogShard(BaseModel):
    """One family shard loaded from ``catalog/by-family/<fid>.json``.

    Contains the family record plus all assets belonging to that
    family.  ``shard_id`` is a UUID unique to each shard file.
    """

    shard_id: str = ""
    updated_at: str = ""
    family: FamilyRecord
    assets: list[AssetRecord] = Field(default_factory=list)


class ShardRef(BaseModel):
    """Lightweight pointer to a shard inside ``manifest.json``."""

    family_id: str
    shard_path: str
    asset_count: int = 0


class CatalogManifest(BaseModel):
    """Top-level manifest listing all family shards.

    Loaded from ``catalog/manifest.json``.
    """

    updated_at: str = ""
    shard_strategy: str = "by_family"
    shards: list[ShardRef] = Field(default_factory=list)


# ── Feature bundle and artifact refs ─────────────────────────────


class ArtifactRef(BaseModel):
    """Pointer from a feature bundle to a heavy artifact (e.g. parquet).

    ``type`` is typically ``"deep_signals"`` for the per-model parquet
    file containing EAS, NLF, LEP, END, WSP, and WVC arrays.
    """

    artifact_id: str
    type: str
    path: str
    signals: list[str] = Field(default_factory=list)
    checksum: str = ""
    availability: str = "local"


class SignalSummary(BaseModel):
    """Scalar metadata about extracted signals (not the arrays themselves).

    Stored in ``features.json`` under the ``signals`` key.  Enables
    quick structural comparisons without loading the parquet.
    """

    hidden_size: int = 0
    num_layers: int = 0
    eas_anchor_count: int = 0
    nlf_mode: str | None = None
    nlf_num_layers: int = 0
    eas_self_sim_shape: list[int] = Field(default_factory=list)
    end_histogram_bins: int = 0
    nlf_vector_length: int = 0
    lep_profile_length: int = 0
    wsp_signature_length: int = 0
    wvc_layer_count: int = 0
    wvc_layer_vector_length: int = 0


class FeatureBundle(BaseModel):
    """Lightweight feature bundle loaded from ``features.json``.

    Contains MFI / TFV metadata, a signal summary, and artifact refs
    pointing to the heavy deep-signals parquet.  The actual numpy
    arrays live in the parquet, not here.
    """

    bundle_id: str = ""
    model_id: str = ""
    asset_id: str = ""
    family_id: str = ""
    source_id: str = ""
    status: str = "ok"
    updated_at: str = ""
    missing_signals: list[str] = Field(default_factory=list)
    mfi: dict[str, Any] = Field(default_factory=dict)
    tfv: dict[str, Any] = Field(default_factory=dict)
    vocab: dict[str, Any] = Field(default_factory=dict)
    signals: SignalSummary = Field(default_factory=SignalSummary)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)

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

"""Pydantic models / data contracts for ProvenanceKit."""

from provenancekit.models.parsing import ModelFormat, TensorMetadata
from provenancekit.models.results import (
    CachedEntry,
    CompareResult,
    LoadResult,
    LoadStrategy,
    PipelineScore,
    ScanMatch,
    ScanMatchScores,
    ScanModelInfo,
    ScanResult,
    ScoreInterpretation,
    SignalScores,
)
from provenancekit.models.signals import (
    MFIFingerprint,
    MFISimilarity,
    TokenizerFeatures,
    VocabOverlap,
    WeightSignalFeatures,
)
from provenancekit.models.storage import (
    ArtifactRef,
    AssetRecord,
    CatalogManifest,
    CatalogShard,
    FamilyRecord,
    FeatureBundle,
    ShardRef,
    SignalSummary,
)

__all__ = [
    "ArtifactRef",
    "AssetRecord",
    "CachedEntry",
    "CatalogManifest",
    "CatalogShard",
    "CompareResult",
    "FamilyRecord",
    "FeatureBundle",
    "LoadResult",
    "LoadStrategy",
    "MFIFingerprint",
    "MFISimilarity",
    "ModelFormat",
    "PipelineScore",
    "ScanMatch",
    "ScanMatchScores",
    "ScanModelInfo",
    "ScanResult",
    "ScoreInterpretation",
    "ShardRef",
    "SignalScores",
    "SignalSummary",
    "TensorMetadata",
    "TokenizerFeatures",
    "VocabOverlap",
    "WeightSignalFeatures",
]

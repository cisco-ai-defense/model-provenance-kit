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

"""Tests for stage 3 verdicts and ranking in provenancekit.core.lookup."""

from collections.abc import Iterator
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from provenancekit.config.constants import EXTRACTOR_VERSION
from provenancekit.core.lookup import (
    _LookupCandidate,
    stage1_param_filter,
    stage3_similarity,
)
from provenancekit.core.scoring import decide
from provenancekit.models.signals import MFISimilarity
from provenancekit.models.storage import FeatureBundle

_LOOKUP = "provenancekit.core.lookup"
_SIGNALS = ("eas", "nlf", "lep", "end", "wvc")


def _candidate(asset_id: str, *, deep_signals: bool = True) -> _LookupCandidate:
    refs = [SimpleNamespace(type="deep_signals", path="p")] if deep_signals else []
    return _LookupCandidate(
        asset=SimpleNamespace(  # type: ignore[arg-type]
            asset_id=asset_id,
            model_id=asset_id,
            family_id="fam",
            param_bucket="<=1B",
        ),
        bundle=SimpleNamespace(  # type: ignore[arg-type]
            mfi={}, tfv={}, artifact_refs=refs, model_id=asset_id, family_id="fam"
        ),
    )


@pytest.fixture
def scoring_patches() -> Iterator[dict[str, Any]]:
    with ExitStack() as stack:
        mocks = {
            "mfi": stack.enter_context(patch(f"{_LOOKUP}.mfi_similarity")),
            "fp": stack.enter_context(patch(f"{_LOOKUP}.MFIFingerprint")),
            "tfv": stack.enter_context(
                patch(f"{_LOOKUP}.tfv_similarity", return_value=1.0)
            ),
        }
        for name in _SIGNALS:
            mocks[name] = stack.enter_context(
                patch(f"{_LOOKUP}.{name}_similarity", return_value=1.0)
            )
        yield mocks


def _db_service() -> MagicMock:
    db = MagicMock()
    db.get_family_display_name.return_value = "Family"
    return db


class TestStage3Verdict:
    def test_tier1_conflict_matches_compare_decision(
        self, scoring_patches: dict[str, Any]
    ) -> None:
        scoring_patches["mfi"].return_value = MFISimilarity(
            score=1.0, tier=1, match_type="exact"
        )
        for name in _SIGNALS:
            scoring_patches[name].return_value = 0.5
        [match] = stage3_similarity(
            MagicMock(), MagicMock(), MagicMock(), [_candidate("db/a")], _db_service()
        )
        expected = decide(1.0, 1, 0.5)
        assert match.provenance_decision == expected.decision
        assert match.scores.verification == "weights_conflict"
        assert match.scores.pipeline_score == 0.5

    def test_tier1_without_deep_signals_unverified(
        self, scoring_patches: dict[str, Any]
    ) -> None:
        scoring_patches["mfi"].return_value = MFISimilarity(
            score=1.0, tier=1, match_type="exact"
        )
        [match] = stage3_similarity(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            [_candidate("db/a", deep_signals=False)],
            _db_service(),
        )
        assert match.provenance_decision == "Metadata Match (unverified)"
        assert match.scores.verification == "unverified"
        assert match.scores.pipeline_score == 1.0


class TestRanking:
    def test_weight_backed_matches_rank_above_unverified(
        self, scoring_patches: dict[str, Any]
    ) -> None:
        from provenancekit.core import lookup

        scoring_patches["mfi"].return_value = MFISimilarity(
            score=1.0, tier=1, match_type="exact"
        )
        for name in _SIGNALS:
            scoring_patches[name].return_value = 0.8
        candidates = [
            _candidate("db/unverified", deep_signals=False),
            _candidate("db/verified"),
        ]
        db = _db_service()
        db.load_catalog.return_value = MagicMock()
        with (
            patch(f"{_LOOKUP}.stage1_param_filter", return_value=candidates),
            patch(f"{_LOOKUP}.stage2_hash_check", return_value=candidates),
        ):
            matches = lookup.run_lookup(
                MagicMock(), MagicMock(), MagicMock(), db, top_k=5, threshold=0.5
            )
        assert [m.asset_id for m in matches] == ["db/verified", "db/unverified"]
        assert matches[0].scores.pipeline_score == 0.8
        assert matches[1].scores.pipeline_score == 1.0

    def test_shallow_query_ranks_by_score(
        self, scoring_patches: dict[str, Any]
    ) -> None:
        """A quantized query has no LEP/WVC against any candidate."""
        from provenancekit.core import lookup

        candidates = [_candidate("db/other"), _candidate("db/base")]
        scoring_patches["mfi"].side_effect = [
            MFISimilarity(score=0.6, tier=3, match_type="none"),
            MFISimilarity(score=1.0, tier=1, match_type="exact"),
        ]
        scoring_patches["lep"].return_value = float("nan")
        scoring_patches["wvc"].return_value = float("nan")
        for name in ("eas", "nlf", "end"):
            scoring_patches[name].side_effect = [0.6, 1.0]
        db = _db_service()
        db.load_catalog.return_value = MagicMock()
        with (
            patch(f"{_LOOKUP}.stage1_param_filter", return_value=candidates),
            patch(f"{_LOOKUP}.stage2_hash_check", return_value=candidates),
        ):
            matches = lookup.run_lookup(
                MagicMock(), MagicMock(), MagicMock(), db, top_k=5, threshold=0.5
            )
        assert [m.asset_id for m in matches] == ["db/base", "db/other"]
        assert matches[0].scores.verification == "weights_shallow"
        assert matches[1].scores.verification == "weights_only"


class TestStage1BundleVersion:
    @pytest.mark.parametrize(
        ("version", "kept"), [(None, True), (EXTRACTOR_VERSION, True), ("1", False)]
    )
    def test_bundle_from_other_version_skipped(
        self, version: str | None, kept: bool
    ) -> None:
        asset = SimpleNamespace(
            asset_id="db/a", param_bucket="<=1B", feature_path="features/a.json"
        )
        catalog = SimpleNamespace(assets={"db/a": asset})
        db = _db_service()
        db.load_feature_bundle.return_value = FeatureBundle(extractor_version=version)
        with patch(f"{_LOOKUP}.compute_param_bucket", return_value="<=1B"):
            candidates = stage1_param_filter(MagicMock(), catalog, db)  # type: ignore[arg-type]
        assert bool(candidates) is kept


class TestStage1UntrustedConfig:
    @pytest.mark.parametrize(("trusted", "kept"), [(True, False), (False, True)])
    def test_bucket_filter_skipped_for_untrusted_config(
        self, trusted: bool, kept: bool
    ) -> None:
        asset = SimpleNamespace(
            asset_id="db/a", param_bucket="40B+", feature_path="features/a.json"
        )
        catalog = SimpleNamespace(assets={"db/a": asset})
        db = _db_service()
        db.load_feature_bundle.return_value = FeatureBundle()
        with patch(f"{_LOOKUP}.compute_param_bucket", return_value="<=1B"):
            candidates = stage1_param_filter(
                MagicMock(),
                catalog,  # type: ignore[arg-type]
                db,
                metadata_trusted=trusted,
            )
        assert bool(candidates) is kept

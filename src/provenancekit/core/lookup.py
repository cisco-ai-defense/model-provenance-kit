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

"""Three-stage database lookup pipeline for model provenance scanning.

Given a test model's extracted features, this module runs a cascading
filter against the provenance database to identify the most likely
base-model match(es).

Pipeline::

    Stage 1  — param_bucket filter   (metadata only, no parquet)
    Stage 2  — arch/family hash check (annotate, does NOT gate)
    Stage 3  — full similarity        (MFI + TFV + weight signals)

    Rank by pipeline_score → threshold → top-K
"""

import math
import time
from dataclasses import dataclass

import structlog

from provenancekit.config.settings import Settings
from provenancekit.core.scoring import (
    compute_identity_score,
    compute_tokenizer_score,
    interpret_score,
)
from provenancekit.core.signals.metadata import similarity as mfi_similarity
from provenancekit.core.signals.tokenizer import tfv_similarity
from provenancekit.core.signals.weight_signals import (
    eas_similarity,
    end_similarity,
    lep_similarity,
    nlf_similarity,
    wvc_similarity,
)
from provenancekit.models.results import ScanMatch, ScanMatchScores
from provenancekit.models.signals import (
    MFIFingerprint,
    TokenizerFeatures,
    WeightSignalFeatures,
)
from provenancekit.models.storage import AssetRecord, FeatureBundle
from provenancekit.services.database import Catalog, DatabaseService
from provenancekit.utils import nan_safe
from provenancekit.utils.tensor import compute_param_bucket

log = structlog.get_logger()

# ── Constants ─────────────────────────────────────────────────────

BUCKETS_ORDER = ["<=1B", "1-10B", "10-40B", "40B+"]

DEFAULT_THRESHOLD = Settings.model_fields["scan_threshold"].default
DEFAULT_TOP_K = Settings.model_fields["scan_top_k"].default


# ── Internal dataclass for pipeline candidates ───────────────────


@dataclass
class _LookupCandidate:
    """A DB asset that survived Stage 1 filtering."""

    asset: AssetRecord
    bundle: FeatureBundle
    hash_match: str | None = None


# ── Helpers ───────────────────────────────────────────────────────


# compute_param_bucket is imported from provenancekit.utils.tensor


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


# ── Stage 1: param_bucket filter ─────────────────────────────────


def stage1_param_filter(
    test_fp: MFIFingerprint,
    catalog: Catalog,
    db_service: DatabaseService,
) -> list[_LookupCandidate]:
    """Filter DB assets by param-bucket proximity.

    Allows the test model's bucket plus one adjacent bucket on each
    side.  For each surviving asset, loads its ``FeatureBundle``.

    Returns:
        Candidates with their feature bundles loaded, ready for
        Stage 2/3.
    """
    t0 = time.perf_counter()
    test_bucket = compute_param_bucket(test_fp)
    try:
        idx = BUCKETS_ORDER.index(test_bucket)
    except ValueError:
        idx = len(BUCKETS_ORDER) - 1
    allowed = set(BUCKETS_ORDER[max(0, idx - 1) : idx + 2])

    log.info(
        "stage1_start",
        test_bucket=test_bucket,
        allowed_buckets=sorted(allowed),
    )

    candidates: list[_LookupCandidate] = []
    skipped = 0

    for asset in catalog.assets.values():
        if asset.param_bucket not in allowed:
            skipped += 1
            continue
        if not asset.feature_path:
            skipped += 1
            continue
        try:
            bundle = db_service.load_feature_bundle(asset.feature_path)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "stage1_bundle_load_failed",
                asset_id=asset.asset_id,
                path=asset.feature_path,
                error=str(exc),
            )
            skipped += 1
            continue
        candidates.append(_LookupCandidate(asset=asset, bundle=bundle))

    log.info(
        "stage1_done",
        candidates=len(candidates),
        filtered_out=skipped,
        elapsed_ms=_elapsed_ms(t0),
    )
    return candidates


# ── Stage 2: hash matching (annotate, don't gate) ────────────────


def stage2_hash_check(
    test_fp: MFIFingerprint,
    candidates: list[_LookupCandidate],
) -> list[_LookupCandidate]:
    """Annotate candidates with arch_hash / family_hash match info.

    All candidates pass through regardless of match — hash information
    is carried forward for logging and result annotation only.  This
    avoids false negatives when config variations (e.g. ``rope_scaling``)
    change the hash despite the model being a clear derivative.
    """
    t0 = time.perf_counter()
    log.info(
        "stage2_start",
        test_arch_hash=test_fp.arch_hash,
        test_family_hash=test_fp.family_hash,
    )

    hash_match_count = 0
    for cand in candidates:
        db_mfi = cand.bundle.mfi
        arch_match = db_mfi.get("arch_hash") == test_fp.arch_hash
        family_match = db_mfi.get("family_hash") == test_fp.family_hash

        if arch_match:
            cand.hash_match = "exact_arch"
            hash_match_count += 1
        elif family_match:
            cand.hash_match = "family_hash"
            hash_match_count += 1

    log.info(
        "stage2_done",
        hash_matches=hash_match_count,
        total_candidates=len(candidates),
        elapsed_ms=_elapsed_ms(t0),
    )

    for cand in candidates:
        if cand.hash_match:
            log.info(
                "stage2_hash_match",
                asset_id=cand.asset.asset_id,
                match_type=cand.hash_match,
            )

    return candidates


# ── Stage 3: full similarity comparison ──────────────────────────


def stage3_similarity(
    test_fp: MFIFingerprint,
    test_tfv: TokenizerFeatures,
    test_ws: WeightSignalFeatures | None,
    candidates: list[_LookupCandidate],
    db_service: DatabaseService,
) -> list[ScanMatch]:
    """Score every candidate using MFI, tokenizer, and weight signals.

    For each candidate:
    1. Compute MFI similarity (tier + score).
    2. Compute TFV similarity.
    3. If the test model has weight signals and the DB model has a
       deep-signals parquet, load it, reconstruct
       ``WeightSignalFeatures``, and compute EAS/NLF/LEP/END/WVC.
    4. Derive ``pipeline_score`` via the MFI gate and ``decision``.

    .. note::
       VOA (Vocabulary Overlap Analysis) is **not** computed here.
       Loading both tokenizers per candidate is too expensive for a full
       database scan.  ``compute_tokenizer_score`` receives ``voa=0.0``
       so the tokenizer score relies on TFV alone.  The pairwise
       ``compare()`` path computes real VOA Jaccard similarity.
    """
    t0 = time.perf_counter()
    log.info("stage3_start", candidates=len(candidates))

    results: list[ScanMatch] = []

    for cand in candidates:
        t_cand = time.perf_counter()
        asset = cand.asset
        bundle = cand.bundle

        # -- MFI similarity --
        db_fp = MFIFingerprint.model_validate(bundle.mfi)
        mfi_sim = mfi_similarity(test_fp, db_fp)

        # -- TFV similarity --
        tfv_s = float("nan")
        try:
            db_tfv = TokenizerFeatures.model_validate(bundle.tfv)
            tfv_s = tfv_similarity(test_tfv, db_tfv)
        except (ValueError, TypeError, KeyError) as exc:
            log.warning(
                "stage3_tfv_parse_failed",
                asset_id=asset.asset_id,
                error=str(exc),
            )

        # -- Weight signal similarities --
        eas_s = float("nan")
        nlf_s = float("nan")
        lep_s = float("nan")
        end_s = float("nan")
        wvc_s = float("nan")

        if test_ws is not None:
            ds_ref = next(
                (r for r in bundle.artifact_refs if r.type == "deep_signals"),
                None,
            )
            if ds_ref:
                deep = db_service.load_deep_signals(ds_ref.path)
                if deep:
                    db_ws = db_service.reconstruct_weight_features(deep, bundle)
                    eas_s = eas_similarity(test_ws, db_ws)
                    nlf_s = nlf_similarity(test_ws, db_ws)
                    lep_s = lep_similarity(test_ws, db_ws)
                    end_s = end_similarity(test_ws, db_ws)
                    wvc_s = wvc_similarity(test_ws, db_ws)

        # -- Composite scores --
        identity = compute_identity_score(eas_s, nlf_s, lep_s, end_s, wvc_s)
        # VOA (vocab_overlap) is omitted in scan mode because it requires
        # loading both tokenizers per candidate, which is prohibitively
        # expensive for large databases.  The compare() path computes real
        # VOA; here we pass 0.0 so tokenizer_score relies on TFV alone.
        tok_score = compute_tokenizer_score(
            tfv_s if not math.isnan(tfv_s) else 0.0,
            0.0,
        )

        # -- MFI gate: decide pipeline_score and decision label --
        if mfi_sim.tier <= 2:
            pipeline_score = mfi_sim.score
            decision = "Confirmed Match"
        elif not math.isnan(identity):
            pipeline_score = identity
            decision = interpret_score(pipeline_score).label
        else:
            # No weight signals available — fall back to MFI soft score
            # so Tier 3 candidates still surface rather than being dropped.
            pipeline_score = mfi_sim.score
            decision = interpret_score(pipeline_score).label + " (metadata only)"

        # -- Derive match_type from hash annotation or MFI tier --
        if cand.hash_match:
            match_type = cand.hash_match
        elif mfi_sim.tier <= 2:
            match_type = "exact_arch" if mfi_sim.tier == 1 else "family_hash"
        elif math.isnan(identity):
            match_type = "mfi_only"
        else:
            match_type = "similarity"

        elapsed = _elapsed_ms(t_cand)

        log.info(
            "stage3_candidate",
            asset_id=asset.asset_id,
            pipeline_score=round(pipeline_score, 4)
            if not math.isnan(pipeline_score)
            else None,
            mfi_score=round(mfi_sim.score, 4),
            mfi_tier=mfi_sim.tier,
            identity_score=nan_safe(identity),
            decision=decision,
            elapsed_ms=elapsed,
        )

        results.append(
            ScanMatch(
                asset_id=asset.asset_id,
                model_id=asset.model_id or bundle.model_id,
                family_id=asset.family_id or bundle.family_id,
                family_name=db_service.get_family_display_name(
                    asset.family_id or bundle.family_id,
                ),
                param_bucket=asset.param_bucket,
                match_type=match_type,
                scores=ScanMatchScores(
                    pipeline_score=nan_safe(pipeline_score),
                    identity_score=nan_safe(identity),
                    mfi_score=mfi_sim.score,
                    mfi_tier=mfi_sim.tier,
                    mfi_match_type=mfi_sim.match_type,
                    tokenizer_score=tok_score,
                    eas=nan_safe(eas_s),
                    nlf=nan_safe(nlf_s),
                    lep=nan_safe(lep_s),
                    end=nan_safe(end_s),
                    wvc=nan_safe(wvc_s),
                    tfv=nan_safe(tfv_s),
                ),
                provenance_decision=decision,
                elapsed_ms=elapsed,
            )
        )

    log.info("stage3_done", elapsed_ms=_elapsed_ms(t0))
    return results


# ── Orchestrator ──────────────────────────────────────────────────


def run_lookup(
    test_fp: MFIFingerprint,
    test_tfv: TokenizerFeatures,
    test_ws: WeightSignalFeatures | None,
    db_service: DatabaseService,
    *,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[ScanMatch]:
    """Run the full 3-stage lookup and return ranked matches.

    Args:
        test_fp: MFI fingerprint of the query model.
        test_tfv: Tokenizer features of the query model.
        test_ws: Weight signal features (may be ``None``).
        db_service: Initialised database service.
        top_k: Maximum number of matches to return.
        threshold: Minimum ``pipeline_score`` for inclusion.

    Returns:
        Up to *top_k* :class:`ScanMatch` objects sorted by descending
        ``pipeline_score``, all above *threshold*.
    """
    t0 = time.perf_counter()
    log.info("lookup_start")

    catalog = db_service.load_catalog()

    # Stage 1 — param bucket filter
    candidates = stage1_param_filter(test_fp, catalog, db_service)
    if not candidates:
        log.warning("lookup_no_candidates_after_stage1")
        return []

    # Stage 2 — hash check (annotate only)
    candidates = stage2_hash_check(test_fp, candidates)

    # Stage 3 — full similarity
    matches = stage3_similarity(test_fp, test_tfv, test_ws, candidates, db_service)

    # Rank, threshold, and trim
    matches.sort(
        key=lambda m: m.scores.pipeline_score or 0,
        reverse=True,
    )

    above = [m for m in matches if (m.scores.pipeline_score or 0) >= threshold][:top_k]

    log.info(
        "lookup_done",
        total_scored=len(matches),
        above_threshold=len(above),
        elapsed_ms=_elapsed_ms(t0),
    )

    for rank, m in enumerate(above, 1):
        log.info(
            "lookup_match",
            rank=rank,
            asset_id=m.asset_id,
            pipeline_score=m.scores.pipeline_score,
            family_id=m.family_id,
            decision=m.provenance_decision,
        )

    return above

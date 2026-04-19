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

"""Model provenance scanner — the main orchestrator.

Coordinates MFI fingerprinting, tokenizer analysis, weight signal
extraction, scoring, and the MFI gate to produce:

*  A ``CompareResult`` for pairwise model comparison.
*  A ``ScanResult`` for one-vs-many database lookup.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog
from huggingface_hub import model_info
from transformers import AutoTokenizer

from provenancekit.config.settings import Settings
from provenancekit.core.lookup import run_lookup
from provenancekit.core.scoring import (
    compute_identity_score,
    compute_tokenizer_score,
    interpret_score,
)
from provenancekit.core.signals.metadata import (
    classify,
    extract_fingerprint,
)
from provenancekit.core.signals.metadata import (
    similarity as mfi_similarity,
)
from provenancekit.core.signals.tokenizer import (
    extract_tokenizer_features,
    tfv_similarity,
    vocab_overlap,
)
from provenancekit.core.signals.weight_signals import (
    eas_similarity,
    end_similarity,
    extract_signals,
    extract_signals_streaming,
    lep_similarity,
    nlf_similarity,
    wvc_similarity,
)
from provenancekit.exceptions import ExtractionError, ModelLoadError
from provenancekit.models.results import (
    CachedEntry,
    CompareResult,
    LoadStrategy,
    PipelineScore,
    ScanModelInfo,
    ScanResult,
    SignalScores,
    TimingBreakdown,
)
from provenancekit.models.signals import (
    MFIFingerprint,
    TokenizerFeatures,
    WeightSignalFeatures,
)
from provenancekit.services.cache import CacheService, NullCache
from provenancekit.services.database import DatabaseService
from provenancekit.services.model_loader import is_local_hf_snapshot, load_state_dict
from provenancekit.utils import nan_safe
from provenancekit.utils.tensor import compute_param_bucket

log = structlog.get_logger()


@dataclass
class _ModelData:
    """Internal bundle of all extracted features for a single model."""

    fp: MFIFingerprint
    tfv: TokenizerFeatures
    ws: WeightSignalFeatures | None
    tokenizer: Any
    base_elapsed: float
    weight_elapsed: float
    mfi_cache_hit: bool
    tfv_cache_hit: bool
    ws_cache_hit: bool


class ModelProvenanceScanner:
    """Concrete provenance scanner implementing the full compare pipeline.

    Orchestrates MFI, TFV, VOA, and weight-level signal extraction,
    scoring, and MFI-gated pipeline decision.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        cache: CacheService | NullCache | None = None,
    ) -> None:
        """Initialise the scanner.

        Args:
            settings: Runtime configuration.  Defaults to ``Settings()``.
            cache: Feature cache.  Pass ``None`` (the default) to use a
                fresh ``CacheService``.  Pass a ``NullCache`` instance
                to disable caching entirely.
        """
        self._settings = settings or Settings()
        if cache is not None:
            self._cache: CacheService | NullCache = cache
        else:
            self._cache = CacheService()
        self._db_service: DatabaseService | None = None

    # ── public API ────────────────────────────────────────────────

    @staticmethod
    def _fetch_param_count(model_id: str) -> int | None:
        """Try to get exact parameter count from HuggingFace Hub metadata.

        Returns ``None`` for local models or on any network/API failure.
        """
        if is_local_hf_snapshot(model_id):
            return None
        try:
            info = model_info(model_id, timeout=5)
            if info.safetensors is not None and info.safetensors.total:
                return info.safetensors.total
        except Exception as exc:  # noqa: BLE001
            log.debug("hub_param_count_failed", model_id=model_id, error=str(exc))
        return None

    def compare(
        self,
        model_a: str,
        model_b: str,
        *,
        on_phase: Callable[[str], None] | None = None,
    ) -> CompareResult:
        """Compare two models and return a detailed provenance result."""
        _phase = on_phase or (lambda _msg: None)

        t0 = time.monotonic()
        log.info("compare_start", model_a=model_a, model_b=model_b)

        _phase("extracting features from model A")
        t_extract_a = time.monotonic()
        log.info("extract_start", model_id=model_a)
        data_a = self._extract_model(model_a)
        extract_a_elapsed = time.monotonic() - t_extract_a
        log.info(
            "extract_done",
            model_id=model_a,
            elapsed=f"{extract_a_elapsed:.1f}s",
        )

        _phase("extracting features from model B")
        t_extract_b = time.monotonic()
        log.info("extract_start", model_id=model_b)
        data_b = self._extract_model(model_b)
        extract_b_elapsed = time.monotonic() - t_extract_b
        log.info(
            "extract_done",
            model_id=model_b,
            elapsed=f"{extract_b_elapsed:.1f}s",
        )

        _phase("computing similarity scores")
        t_score = time.monotonic()
        log.info("scoring_start")
        mfi = mfi_similarity(data_a.fp, data_b.fp)
        family_a, _ = classify(data_a.fp)
        family_b, _ = classify(data_b.fp)

        tfv_s = tfv_similarity(data_a.tfv, data_b.tfv)

        voa = vocab_overlap(
            model_a,
            model_b,
            tok_a=data_a.tokenizer,
            tok_b=data_b.tokenizer,
        )
        voa_s = voa.jaccard

        eas_s = eas_similarity(data_a.ws, data_b.ws)
        nlf_s = nlf_similarity(data_a.ws, data_b.ws)
        lep_s = lep_similarity(data_a.ws, data_b.ws)
        end_s = end_similarity(data_a.ws, data_b.ws)
        wvc_s = wvc_similarity(data_a.ws, data_b.ws)

        identity = compute_identity_score(eas_s, nlf_s, lep_s, end_s, wvc_s)
        tokenizer = compute_tokenizer_score(tfv_s, voa_s)

        if mfi.tier <= 2:
            pipeline_score = mfi.score
            decision = "same_family"
        elif not math.isnan(identity):
            pipeline_score = identity
            decision = ""
        else:
            pipeline_score = mfi.score
            decision = interpret_score(pipeline_score).label + " (metadata only)"

        interpretation = interpret_score(pipeline_score)
        if not decision:
            decision = interpretation.label
        scoring_elapsed = time.monotonic() - t_score
        metadata_extract_elapsed = data_a.base_elapsed + data_b.base_elapsed
        weight_feature_extract_elapsed = data_a.weight_elapsed + data_b.weight_elapsed
        cache_hit_tokens: list[str] = []
        if data_a.ws_cache_hit or data_b.ws_cache_hit:
            cache_hit_tokens.append("ws")
        if data_a.mfi_cache_hit or data_b.mfi_cache_hit:
            cache_hit_tokens.append("mfi")
        if data_a.tfv_cache_hit or data_b.tfv_cache_hit:
            cache_hit_tokens.append("tfv")
        cache_hit = ", ".join(cache_hit_tokens) if cache_hit_tokens else "False"
        log.info("scoring_done", elapsed=f"{scoring_elapsed:.1f}s")
        elapsed = time.monotonic() - t0

        log.info(
            "compare_done",
            pipeline_score=pipeline_score,
            mfi_tier=mfi.tier,
            decision=decision,
            elapsed=f"{elapsed:.1f}s",
            metadata_extract_elapsed=f"{metadata_extract_elapsed:.1f}s",
            weight_feature_extract_elapsed=f"{weight_feature_extract_elapsed:.1f}s",
            cache_hit=cache_hit,
            scoring_elapsed=f"{scoring_elapsed:.1f}s",
        )

        return CompareResult(
            model_a=model_a,
            model_b=model_b,
            family_a=family_a,
            family_b=family_b,
            signals=SignalScores(
                eas=nan_safe(eas_s),
                nlf=nan_safe(nlf_s),
                lep=nan_safe(lep_s),
                end=nan_safe(end_s),
                wvc=nan_safe(wvc_s),
                tfv=nan_safe(tfv_s),
                voa=nan_safe(voa_s),
            ),
            scores=PipelineScore(
                mfi_score=mfi.score,
                mfi_tier=mfi.tier,
                mfi_match=mfi.match_type,
                identity_score=nan_safe(identity),
                tokenizer_score=nan_safe(tokenizer),
                pipeline_score=pipeline_score,
                provenance_decision=decision,
            ),
            interpretation=interpretation,
            time_seconds=round(elapsed, 3),
            timing=TimingBreakdown(
                total_seconds=round(elapsed, 3),
                metadata_extract_seconds=round(metadata_extract_elapsed, 3),
                weight_feature_extract_seconds=round(weight_feature_extract_elapsed, 3),
                cache_hit=cache_hit,
            ),
        )

    def scan(
        self,
        model_id: str,
        *,
        top_k: int | None = None,
        threshold: float | None = None,
        on_phase: Callable[[str], None] | None = None,
    ) -> ScanResult:
        """Scan a model against the provenance database.

        Extracts features from *model_id*, then runs a 3-stage lookup
        (param-bucket filter → hash check → full similarity) against
        every asset in the seed database.

        Args:
            model_id: HuggingFace model identifier or local path.
            top_k: Maximum matches to return.  Falls back to
                ``settings.scan_top_k``.
            threshold: Minimum ``pipeline_score`` for inclusion.
                Falls back to ``settings.scan_threshold``.
            on_phase: Optional callback invoked with a short description
                when the scan transitions between phases.

        Returns:
            A :class:`ScanResult` containing query model metadata and
            up to *top_k* ranked :class:`ScanMatch` entries.
        """
        _phase = on_phase or (lambda _msg: None)

        effective_top_k = top_k if top_k is not None else self._settings.scan_top_k
        effective_threshold = (
            threshold if threshold is not None else self._settings.scan_threshold
        )

        t0 = time.monotonic()
        log.info(
            "scan_start",
            model_id=model_id,
            top_k=effective_top_k,
            threshold=effective_threshold,
        )

        # 1. Extract features from the query model (reuses compare's extraction)
        _phase("extracting features")
        t_extract = time.monotonic()
        data = self._extract_model(model_id)
        extract_seconds = round(time.monotonic() - t_extract, 3)

        num_params = self._fetch_param_count(model_id)
        log.info(
            "param_count",
            model_id=model_id,
            source="hub" if num_params is not None else "estimate",
        )

        # 2. Run 3-stage DB lookup
        _phase("matching against provenance database")
        t_lookup = time.monotonic()
        db_service = self._get_db_service()
        matches = run_lookup(
            data.fp,
            data.tfv,
            data.ws,
            db_service,
            top_k=effective_top_k,
            threshold=effective_threshold,
        )
        lookup_seconds = round(time.monotonic() - t_lookup, 3)

        # 3. Build ScanResult
        _phase("building results")
        elapsed_ms = round((time.monotonic() - t0) * 1000, 2)

        model_info = ScanModelInfo(
            model_path=model_id,
            model_type=data.fp.model_type,
            architectures=data.fp.architectures,
            hidden_size=data.fp.hidden_size,
            num_hidden_layers=data.fp.num_hidden_layers,
            num_attention_heads=data.fp.num_attention_heads,
            num_key_value_heads=data.fp.num_key_value_heads,
            head_dim=data.fp.head_dim,
            intermediate_size=data.fp.intermediate_size,
            vocab_size=data.fp.vocab_size,
            tie_word_embeddings=data.fp.tie_word_embeddings,
            hidden_act=data.fp.hidden_act,
            num_parameters=num_params,
            arch_hash=data.fp.arch_hash,
            family_hash=data.fp.family_hash,
            param_bucket=compute_param_bucket(data.fp),
            has_weight_signals=data.ws is not None,
        )

        result = ScanResult(
            model_info=model_info,
            matches=matches,
            match_count=len(matches),
            elapsed_ms=elapsed_ms,
            extract_seconds=extract_seconds,
            lookup_seconds=lookup_seconds,
        )

        log.info(
            "scan_done",
            model_id=model_id,
            match_count=result.match_count,
            elapsed_ms=elapsed_ms,
        )

        return result

    # ── private helpers ───────────────────────────────────────────

    def _get_db_service(self) -> DatabaseService:
        """Return the (lazily created) database service."""
        if self._db_service is None:
            self._db_service = DatabaseService(self._settings.db_root)
        return self._db_service

    def _extract_model(self, model_id: str) -> _ModelData:
        """Extract all features for a single model, with caching."""
        model_id = model_id.strip()
        cached = self._cache.get(model_id)

        mfi_cache_hit = cached is not None and cached.mfi is not None
        tfv_cache_hit = cached is not None and cached.tfv is not None
        ws_cache_hit = cached is not None and cached.ws is not None

        t_base = time.monotonic()
        fp, tfv, tokenizer = self._extract_base(model_id, cached)
        base_elapsed = time.monotonic() - t_base

        t_ws = time.monotonic()
        ws: WeightSignalFeatures | None
        if ws_cache_hit and cached is not None and cached.ws is not None:
            log.info("cache_hit_ws", model_id=model_id)
            ws = WeightSignalFeatures.from_cache_dict(cached.ws)
        else:
            ws = self._extract_weight_signals(model_id, tokenizer)
            if ws is not None:
                self._persist_ws(model_id, ws)
        ws_elapsed = time.monotonic() - t_ws

        log.info(
            "model_extract_done",
            model_id=model_id,
            base_elapsed=f"{base_elapsed:.1f}s",
            weight_elapsed=f"{ws_elapsed:.1f}s",
            weight_available=ws is not None,
            mfi_cache_hit=mfi_cache_hit,
            tfv_cache_hit=tfv_cache_hit,
            ws_cache_hit=ws_cache_hit,
        )

        return _ModelData(
            fp=fp,
            tfv=tfv,
            ws=ws,
            tokenizer=tokenizer,
            base_elapsed=base_elapsed,
            weight_elapsed=ws_elapsed,
            mfi_cache_hit=mfi_cache_hit,
            tfv_cache_hit=tfv_cache_hit,
            ws_cache_hit=ws_cache_hit,
        )

    def _extract_base(
        self,
        model_id: str,
        cached: CachedEntry | None = None,
    ) -> tuple[MFIFingerprint, TokenizerFeatures, Any]:
        """Extract MFI fingerprint and tokenizer features, using cache.

        Always returns a live tokenizer reference so that downstream
        weight extraction (EAS anchor selection) works correctly even
        on cache hits.

        Raises:
            ExtractionError: When fingerprint or tokenizer extraction
                fails for the given model.
        """
        if cached is not None and cached.mfi is not None and cached.tfv is not None:
            fp = MFIFingerprint.model_validate(cached.mfi)
            tfv = TokenizerFeatures.model_validate(cached.tfv)
            log.debug("cache_hit", model_id=model_id, layer="base")
            tokenizer = self._load_tokenizer_for_cache_hit(
                model_id, self._settings.trust_remote_code
            )
            return fp, tfv, tokenizer

        try:
            log.info("fingerprint_start", model_id=model_id)
            fp, tokenizer = extract_fingerprint(
                model_id,
                trust_remote_code=self._settings.trust_remote_code,
            )
            log.info("fingerprint_done", model_id=model_id)
            log.info("tfv_start", model_id=model_id)
            tfv = extract_tokenizer_features(
                model_id,
                tokenizer=tokenizer,
                trust_remote_code=self._settings.trust_remote_code,
            )
            log.info("tfv_done", model_id=model_id)
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(
                f"Failed to extract base features for '{model_id}': {exc}",
                details={"model_id": model_id, "stage": "base_extraction"},
            ) from exc

        entry = CachedEntry(
            model_id=model_id,
            mfi=fp.model_dump(),
            tfv=tfv.model_dump(),
            ws=cached.ws if cached is not None else None,
        )
        self._cache.put(model_id, entry)
        return fp, tfv, tokenizer

    @staticmethod
    def _load_tokenizer_for_cache_hit(
        model_id: str, trust_remote_code: bool = False
    ) -> Any:
        """Load a tokenizer when base features came from cache."""
        try:
            return AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            log.debug(
                "tokenizer_reload_failed",
                model_id=model_id,
                error=str(exc),
            )
            return None

    def _persist_ws(self, model_id: str, ws: WeightSignalFeatures) -> None:
        """Merge weight-signal features into the existing cache entry."""
        cached = self._cache.get(model_id)
        if cached is None:
            cached = CachedEntry(model_id=model_id)
        cached.ws = ws.to_cache_dict()
        self._cache.put(model_id, cached)
        log.info("cache_stored_ws", model_id=model_id)

    @staticmethod
    def _get_vocab(tokenizer: Any) -> list[str] | None:
        if tokenizer is None:
            return None
        try:
            return list(tokenizer.get_vocab().keys())
        except Exception:  # noqa: BLE001
            return None

    def _extract_weight_signals(
        self, model_id: str, tokenizer: Any
    ) -> WeightSignalFeatures | None:
        """Extract weight-level signals, choosing full vs streaming.

        Returns ``None`` when the model cannot be loaded or signals
        cannot be extracted — the comparison continues without weight
        signals in that case.
        """
        log.info("weight_load_start", model_id=model_id)
        t_load = time.monotonic()
        try:
            result = load_state_dict(model_id, settings=self._settings)
        except ModelLoadError as exc:
            log.warning("model_load_failed", model_id=model_id, error=str(exc))
            return None
        load_elapsed = time.monotonic() - t_load

        log.info(
            "weight_load_done",
            model_id=model_id,
            strategy=result.strategy,
            source=result.source,
            elapsed=f"{load_elapsed:.1f}s",
        )

        vocab = self._get_vocab(tokenizer)

        try:
            t_weight_compute = time.monotonic()
            if result.strategy == LoadStrategy.full and result.state_dict is not None:
                log.info("weight_signals_start", model_id=model_id, mode="full")
                ws = extract_signals(
                    result.state_dict,
                    result.config,
                    tokenizer=tokenizer,
                    vocab=vocab,
                    settings=self._settings,
                )
                log.info(
                    "weight_signals_done",
                    model_id=model_id,
                    elapsed=f"{time.monotonic() - t_weight_compute:.1f}s",
                )
                return ws

            if result.strategy == LoadStrategy.streaming:
                log.info("weight_signals_start", model_id=model_id, mode="streaming")
                ws = extract_signals_streaming(
                    model_id,
                    tokenizer=tokenizer,
                    vocab=vocab,
                    settings=self._settings,
                )
                log.info(
                    "weight_signals_done",
                    model_id=model_id,
                    elapsed=f"{time.monotonic() - t_weight_compute:.1f}s",
                )
                return ws
        except ExtractionError as exc:
            log.warning("signal_extraction_failed", model_id=model_id, error=str(exc))
            return None

        log.warning(
            "no_weight_signals_available",
            model_id=model_id,
            source=result.source,
        )
        return None

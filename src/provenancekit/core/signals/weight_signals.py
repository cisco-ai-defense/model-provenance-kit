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

"""Weight-level signal extraction and similarity (EAS, NLF, LEP, END, WVC, WSP).

Two public extraction entry points:

* :func:`extract_signals` — operates on a fully-loaded ``state_dict``.
* :func:`extract_signals_streaming` — lazy per-tensor I/O for large models.

Six public similarity functions compute NaN-aware pairwise scores from
:class:`WeightSignalFeatures` objects.
"""

import gc
import os
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import structlog
import torch
from transformers import AutoConfig

from provenancekit.config.settings import Settings
from provenancekit.core.signals.anchors import get_anchor_ids
from provenancekit.core.signals.config_consistency import (
    check_config_consistency,
    collect_shapes,
)
from provenancekit.core.signals.streamers import create_streamer
from provenancekit.exceptions import ExtractionError
from provenancekit.models.signals import ConfigInconsistency, WeightSignalFeatures
from provenancekit.services.model_loader import (
    is_local_hf_snapshot,
    resolve_hf_model_ref,
    try_hf_download,
)
from provenancekit.utils.paths import UnsafeShardPathError, safe_shard_join
from provenancekit.utils.tensor import (
    canonical_tensor_key,
    classify_tensor_name,
    cosine_clamp,
    extract_layer_index,
    find_embedding_in_state_dict,
    find_embedding_name_in_weight_map,
    is_norm_tensor_name,
    norm_vector_to_stats,
)

log = structlog.get_logger()

_LIGHT_SAMPLE_SIZE: int = 131072  # 128K elements
_MAX_EMBEDDING_ROWS: int = 1_000_000  # blocks adversarial shapes
_SLICE_DISK_THRESHOLD_BYTES: float = 20e9  # row-slice LEP/WSP above this size


def _to_float(t: torch.Tensor) -> torch.Tensor:
    """Promote to float32 when the dtype is not numpy-compatible (e.g. bfloat16)."""
    if t.is_floating_point() and t.dtype not in (torch.bfloat16, torch.float16):
        return t
    return t.float()


# ── Public extraction API ──────────────────────────────────────────


def extract_signals(
    state_dict: dict[str, Any],
    config: Any | None,
    tokenizer: Any | None = None,
    vocab: set[str] | list[str] | None = None,
    mode: str = "deep",
    settings: Settings | None = None,
) -> WeightSignalFeatures | None:
    """Extract weight-level signals from a fully-loaded state_dict.

    Called when ``LoadResult.strategy == "full"``.
    """
    return _extract_signals_impl(state_dict, config, tokenizer, vocab, mode, settings)


@torch.inference_mode()
def _extract_signals_impl(
    state_dict: dict[str, Any],
    config: Any | None,
    tokenizer: Any | None = None,
    vocab: set[str] | list[str] | None = None,
    mode: str = "deep",
    settings: Settings | None = None,
) -> WeightSignalFeatures | None:
    if settings is None:
        settings = Settings()
    light = mode == "light"

    hidden_size = getattr(config, "hidden_size", 0) if config else 0
    num_layers = getattr(config, "num_hidden_layers", 0) if config else 0

    log.info("signal_eas_end_start", n_tensors=len(state_dict))
    eas_self_sim, eas_anchor_count, end_histogram = _extract_eas_end(
        state_dict,
        tokenizer,
        vocab,
    )
    log.info("signal_eas_end_done", anchor_count=eas_anchor_count)

    log.info("signal_nlf_start")
    nlf_vector, nlf_mode, nlf_num_layers = _extract_nlf(state_dict)
    log.info("signal_nlf_done", mode=nlf_mode, layers=nlf_num_layers)

    log.info("signal_lep_wsp_wvc_start")
    lep_profile, wsp_signature, wvc_layer_sigs = _extract_lep_wsp_wvc(
        state_dict,
        light,
        settings.wvc_subsample,
    )
    log.info("signal_lep_wsp_wvc_done")

    config_inconsistencies = _safe_config_check(config, collect_shapes(state_dict))

    gc.collect()

    return WeightSignalFeatures(
        hidden_size=hidden_size,
        num_layers=num_layers,
        eas_self_sim=eas_self_sim,
        eas_anchor_count=eas_anchor_count,
        nlf_vector=nlf_vector,
        nlf_mode=nlf_mode,
        nlf_num_layers=nlf_num_layers,
        lep_profile=lep_profile,
        end_histogram=end_histogram,
        wsp_signature=wsp_signature,
        wvc_layer_sigs=wvc_layer_sigs,
        config_inconsistencies=config_inconsistencies,
    )


def extract_signals_streaming(
    model_name: str,
    tokenizer: Any | None = None,
    vocab: set[str] | list[str] | None = None,
    mode: str = "deep",
    settings: Settings | None = None,
) -> WeightSignalFeatures | None:
    """Extract weight-level signals via streaming for large models.

    Called when ``LoadResult.strategy == "streaming"``. Delegates all
    format-specific I/O to a :class:`~streamers.TensorStreamer` backend
    (safetensors or PyTorch), keeping peak memory at ~1-2 GB regardless
    of model size.
    """
    return _extract_signals_streaming_impl(
        model_name,
        tokenizer,
        vocab,
        mode,
        settings,
    )


@torch.inference_mode()
def _extract_signals_streaming_impl(
    model_name: str,
    tokenizer: Any | None = None,
    vocab: set[str] | list[str] | None = None,
    mode: str = "deep",
    settings: Settings | None = None,
) -> WeightSignalFeatures | None:
    if settings is None:
        settings = Settings()
    model_name = resolve_hf_model_ref(model_name.strip())

    streamer = create_streamer(model_name)
    weight_map = streamer.weight_map
    log.info(
        "streaming_format_resolved", model_id=model_name, format=streamer.format.value
    )

    config: Any | None = None
    config_exc: Exception | None = None
    for local_only in (True, False):
        try:
            log.info(
                "streaming_config_probe",
                model_id=model_name,
                local_only=local_only,
            )
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=settings.trust_remote_code,
                local_files_only=local_only,
            )
            break
        except Exception as exc:  # noqa: BLE001
            config_exc = exc
    if config is None:
        raise ExtractionError(
            f"Failed to load config for streaming extraction of "
            f"'{model_name}': {config_exc}",
            details={"model_id": model_name, "stage": "config_load"},
        ) from config_exc

    hidden_size: int = getattr(config, "hidden_size", 0)
    num_layers: int = getattr(config, "num_hidden_layers", 0)

    shard_files = sorted(set(weight_map.values()))
    log.info(
        "streaming_start",
        model_id=model_name,
        n_shards=len(shard_files),
        n_tensors=len(weight_map),
    )
    _shard_paths: dict[str, str] = {}

    def _get_shard_path(shard_file: str) -> str:
        if shard_file not in _shard_paths:
            if is_local_hf_snapshot(model_name):
                try:
                    _shard_paths[shard_file] = safe_shard_join(model_name, shard_file)
                except UnsafeShardPathError as exc:
                    raise ExtractionError(
                        str(exc),
                        details={"model_name": model_name, "shard": shard_file},
                    ) from exc
            else:
                path = try_hf_download(model_name, shard_file)
                if path is None:
                    raise ExtractionError(
                        f"Failed to resolve shard '{shard_file}' for '{model_name}'",
                        details={
                            "model_id": model_name,
                            "stage": "shard_download",
                            "shard": shard_file,
                        },
                    )
                _shard_paths[shard_file] = path
        return _shard_paths[shard_file]

    def _load_tensor(tensor_name: str) -> Any:
        shard_file = weight_map[tensor_name]
        path = _get_shard_path(shard_file)
        with streamer.open_shard(path) as handle:
            return streamer.get_tensor(handle, tensor_name)

    # ── EAS + END ──
    log.info("streaming_eas_end_start", model_id=model_name)
    eas_self_sim: np.ndarray | None = None
    eas_anchor_count = 0
    end_histogram: np.ndarray | None = None
    tensor_shapes: dict[str, tuple[int, ...]] = {}

    emb_name = find_embedding_name_in_weight_map(weight_map)
    if emb_name is not None:
        try:
            emb_tensor = _to_float(_load_tensor(emb_name))
            tensor_shapes[emb_name] = tuple(emb_tensor.shape)
            v_size, _ = emb_tensor.shape
            if v_size > _MAX_EMBEDDING_ROWS:
                log.warning(
                    "embedding_too_large",
                    vocab_size=v_size,
                    limit=_MAX_EMBEDDING_ROWS,
                )
                del emb_tensor
                raise ValueError("embedding exceeds row limit")
            anchor_ids = get_anchor_ids(tokenizer, vocab, v_size)

            if len(anchor_ids) >= 16:
                anchor_emb = emb_tensor[anchor_ids]
                norms = torch.norm(anchor_emb, dim=1, keepdim=True).clamp(
                    min=1e-8,
                )
                normed = anchor_emb / norms
                eas_self_sim = (normed @ normed.T).cpu().numpy()
                eas_anchor_count = len(anchor_ids)

                all_norms = torch.norm(emb_tensor, dim=1).cpu().numpy()
                hist, _ = np.histogram(all_norms, bins=20, density=True)
                end_histogram = hist / (hist.sum() + 1e-10)

            del emb_tensor
        except Exception as exc:  # noqa: BLE001
            log.debug("streaming_eas_failed", model_id=model_name, error=str(exc))

    log.info(
        "streaming_eas_end_done", model_id=model_name, anchor_count=eas_anchor_count
    )

    # ── NLF ──
    log.info("streaming_nlf_start", model_id=model_name)
    named_norms: list[tuple[str, np.ndarray]] = []
    for tname in weight_map:
        if not is_norm_tensor_name(tname):
            continue
        try:
            t = _to_float(_load_tensor(tname))
            if t.dim() != 1 or t.numel() < 64:
                del t
                continue
            named_norms.append((tname, t.cpu().numpy()))
            del t
        except Exception as exc:  # noqa: BLE001
            log.debug("streaming_nlf_tensor_failed", tensor=tname, error=str(exc))
            continue

    nlf_vector, nlf_mode, nlf_num_layers = _build_nlf(named_norms)

    log.info(
        "streaming_nlf_done", model_id=model_name, mode=nlf_mode, layers=nlf_num_layers
    )

    # ── LEP + WSP + WVC: shard-by-shard scan ──
    log.info("streaming_lep_wsp_wvc_start", model_id=model_name)
    total_disk_bytes = sum(
        os.path.getsize(_get_shard_path(sf))  # noqa: PTH202
        for sf in shard_files
        if os.path.isfile(_get_shard_path(sf))  # noqa: PTH113
    )
    _use_slicing = (
        total_disk_bytes > _SLICE_DISK_THRESHOLD_BYTES and streamer.supports_slicing
    )
    _slice_sample_rows = 32

    light = (mode == "light") or _use_slicing
    layer_energy: dict[int, float] = defaultdict(float)
    layer_signs: dict[int, dict[str, int]] = defaultdict(
        lambda: {"positive": 0, "near_zero": 0, "total": 0},
    )
    wvc_layer_data: dict[int, list[tuple[str, np.ndarray]]] = defaultdict(list)

    shard_to_tensors: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for tname, shard_file in weight_map.items():
        layer_idx = extract_layer_index(tname)
        if layer_idx is not None:
            shard_to_tensors[shard_file].append((tname, layer_idx))

    n_shards_to_scan = len(shard_to_tensors)
    for shard_idx, shard_file in enumerate(
        sorted(shard_to_tensors.keys()),
        1,
    ):
        tensors_in_shard = shard_to_tensors[shard_file]
        if not tensors_in_shard:
            continue
        log.info(
            "streaming_shard",
            shard=f"{shard_idx}/{n_shards_to_scan}",
            file=shard_file,
            n_tensors=len(tensors_in_shard),
        )
        try:
            path = _get_shard_path(shard_file)
            with streamer.open_shard(path) as handle:
                for tname, layer_idx in tensors_in_shard:
                    try:
                        if _use_slicing:
                            tensor_shapes[tname] = tuple(
                                streamer.get_slice(handle, tname).get_shape()
                            )
                            _process_tensor_sliced(
                                handle,
                                tname,
                                layer_idx,
                                _slice_sample_rows,
                                settings.wvc_subsample,
                                layer_energy,
                                layer_signs,
                                wvc_layer_data,
                            )
                        else:
                            tensor = _to_float(streamer.get_tensor(handle, tname))
                            tensor_shapes[tname] = tuple(tensor.shape)
                            _process_tensor_full(
                                tensor,
                                tname,
                                layer_idx,
                                light,
                                settings.wvc_subsample,
                                layer_energy,
                                layer_signs,
                                wvc_layer_data,
                            )
                            del tensor
                    except Exception as exc:  # noqa: BLE001
                        log.debug(
                            "streaming_tensor_process_failed",
                            tensor=tname,
                            error=str(exc),
                        )
                        continue
            if shard_idx % 5 == 0:
                gc.collect()
        except Exception as exc:  # noqa: BLE001
            log.debug("streaming_shard_failed", shard=shard_file, error=str(exc))
            continue

    log.info("streaming_lep_wsp_wvc_done", model_id=model_name)
    lep_profile = _build_lep(layer_energy)
    wsp_signature = _build_wsp(layer_signs)
    wvc_layer_sigs = _build_wvc(wvc_layer_data)
    config_inconsistencies = _safe_config_check(config, tensor_shapes)

    gc.collect()
    log.info("streaming_complete", model_id=model_name)
    return WeightSignalFeatures(
        hidden_size=hidden_size,
        num_layers=num_layers,
        eas_self_sim=eas_self_sim,
        eas_anchor_count=eas_anchor_count,
        nlf_vector=nlf_vector,
        nlf_mode=nlf_mode,
        nlf_num_layers=nlf_num_layers,
        lep_profile=lep_profile,
        end_histogram=end_histogram,
        wsp_signature=wsp_signature,
        wvc_layer_sigs=wvc_layer_sigs,
        config_inconsistencies=config_inconsistencies,
    )


def _safe_config_check(
    config: Any | None, shapes: dict[str, tuple[int, ...]]
) -> list[ConfigInconsistency] | None:
    """Run the config/shape check; ``None`` means it could not be run."""
    if config is None or not shapes:
        return None
    try:
        return check_config_consistency(config, shapes)
    except Exception as exc:  # noqa: BLE001
        log.warning("config_consistency_check_failed", error=str(exc))
        return None


# ── Similarity functions (NaN-aware) ──────────────────────────────


def eas_similarity(
    a: WeightSignalFeatures | None,
    b: WeightSignalFeatures | None,
) -> float:
    """Embedding Anchor Similarity: correlation of self-similarity matrices."""
    if a is None or b is None:
        return float("nan")
    sa, sb = a.eas_self_sim, b.eas_self_sim
    if sa is None or sb is None:
        return float("nan")

    k = min(sa.shape[0], sb.shape[0])
    if k < 8:
        return float("nan")

    sa, sb = sa[:k, :k], sb[:k, :k]
    idx = np.triu_indices(k, k=1)
    ua, ub = sa[idx], sb[idx]

    if ua.std() < 1e-8 or ub.std() < 1e-8:
        return float("nan")

    corr = float(np.corrcoef(ua, ub)[0, 1])
    if np.isnan(corr):
        return float("nan")
    return round((corr + 1.0) / 2.0, 4)


def nlf_similarity(
    a: WeightSignalFeatures | None,
    b: WeightSignalFeatures | None,
) -> float:
    """Norm Layer Fingerprint similarity."""
    if a is None or b is None:
        return float("nan")
    va, vb = a.nlf_vector, b.nlf_vector
    if va is None or vb is None:
        return float("nan")

    mode_a = a.nlf_mode or "stats"
    mode_b = b.nlf_mode or "stats"

    if mode_a == "direct" and mode_b == "direct" and len(va) == len(vb):
        return cosine_clamp(va, vb)

    if mode_a == "direct" and mode_b == "direct" and len(va) != len(vb):
        va = norm_vector_to_stats(va, a.nlf_num_layers)
        vb = norm_vector_to_stats(vb, b.nlf_num_layers)

    if mode_a == "direct" and mode_b != "direct":
        va = norm_vector_to_stats(va, a.nlf_num_layers)
    if mode_b == "direct" and mode_a != "direct":
        vb = norm_vector_to_stats(vb, b.nlf_num_layers)

    min_len = min(len(va), len(vb))
    if min_len < 4:
        return float("nan")
    return cosine_clamp(va[:min_len], vb[:min_len])


def lep_similarity(
    a: WeightSignalFeatures | None,
    b: WeightSignalFeatures | None,
) -> float:
    """Layer Energy Profile similarity: Pearson correlation of energy curves."""
    if a is None or b is None:
        return float("nan")
    pa, pb = a.lep_profile, b.lep_profile
    if pa is None or pb is None:
        return float("nan")

    la, lb = len(pa), len(pb)
    if la == 0 or lb == 0:
        return float("nan")

    if la != lb:
        target = min(la, lb)
        x_target = np.linspace(0, 1, target)
        pa = np.interp(x_target, np.linspace(0, 1, la), pa)
        pb = np.interp(x_target, np.linspace(0, 1, lb), pb)

    if pa.std() < 1e-8 or pb.std() < 1e-8:
        return float("nan")

    corr = float(np.corrcoef(pa, pb)[0, 1])
    if np.isnan(corr):
        return float("nan")
    return round((corr + 1.0) / 2.0, 4)


def end_similarity(
    a: WeightSignalFeatures | None,
    b: WeightSignalFeatures | None,
) -> float:
    """Embedding Norm Distribution similarity: cosine of histograms."""
    if a is None or b is None:
        return float("nan")
    ha, hb = a.end_histogram, b.end_histogram
    if ha is None or hb is None:
        return float("nan")
    return cosine_clamp(ha, hb)


def wvc_similarity(
    a: WeightSignalFeatures | None,
    b: WeightSignalFeatures | None,
) -> float:
    """Weight-Value Cosine: mean layer-by-layer cosine of weight signatures."""
    if a is None or b is None:
        return float("nan")
    sigs_a, sigs_b = a.wvc_layer_sigs, b.wvc_layer_sigs
    if sigs_a is None or sigs_b is None:
        return float("nan")

    common = sorted(set(sigs_a.keys()) & set(sigs_b.keys()))
    if len(common) < 2:
        return float("nan")

    cos_sims: list[float] = []
    for li in common:
        sa, sb = sigs_a[li], sigs_b[li]
        ml = min(len(sa), len(sb))
        if ml < 64:
            continue
        sa, sb = sa[:ml], sb[:ml]
        norm = float(np.linalg.norm(sa) * np.linalg.norm(sb))
        if norm < 1e-10:
            continue
        cos = float(np.dot(sa, sb) / norm)
        cos_sims.append(max(0.0, min(1.0, cos)))

    if len(cos_sims) < 2:
        return float("nan")
    return round(float(np.mean(cos_sims)), 4)


def wsp_similarity(
    a: WeightSignalFeatures | None,
    b: WeightSignalFeatures | None,
) -> float:
    """Weight Sign Pattern similarity: cosine of signature vectors."""
    if a is None or b is None:
        return float("nan")
    sa, sb = a.wsp_signature, b.wsp_signature
    if sa is None or sb is None:
        return float("nan")

    ml = min(len(sa), len(sb))
    if ml < 4:
        return float("nan")
    return cosine_clamp(sa[:ml], sb[:ml])


# ── Private extraction helpers ─────────────────────────────────────


def _extract_eas_end(
    state_dict: dict[str, Any],
    tokenizer: Any | None,
    vocab: set[str] | list[str] | None,
) -> tuple[np.ndarray | None, int, np.ndarray | None]:
    """Extract EAS self-similarity matrix and END histogram."""
    emb_tensor = find_embedding_in_state_dict(state_dict)
    if emb_tensor is None:
        return None, 0, None

    emb = emb_tensor.detach().float()
    v_size, _ = emb.shape
    if v_size > _MAX_EMBEDDING_ROWS:
        log.warning(
            "embedding_too_large",
            vocab_size=v_size,
            limit=_MAX_EMBEDDING_ROWS,
        )
        return None, 0, None
    anchor_ids = get_anchor_ids(tokenizer, vocab, v_size)

    if len(anchor_ids) < 16:
        return None, 0, None

    anchor_emb = emb[anchor_ids]
    norms = torch.norm(anchor_emb, dim=1, keepdim=True).clamp(min=1e-8)
    normed = anchor_emb / norms
    eas_self_sim: np.ndarray = (normed @ normed.T).cpu().numpy()

    all_norms = torch.norm(emb, dim=1).cpu().numpy()
    hist, _ = np.histogram(all_norms, bins=20, density=True)
    end_histogram: np.ndarray = hist / (hist.sum() + 1e-10)

    return eas_self_sim, len(anchor_ids), end_histogram


def _extract_nlf(
    state_dict: dict[str, Any],
) -> tuple[np.ndarray | None, str | None, int]:
    """Extract NLF vector from norm-layer weights."""
    named_norms: list[tuple[str, np.ndarray]] = [
        (name, param.detach().float().cpu().numpy())
        for name, param in state_dict.items()
        if is_norm_tensor_name(name) and param.dim() == 1 and param.numel() >= 64
    ]
    return _build_nlf(named_norms)


def _build_nlf(
    named_norms: list[tuple[str, np.ndarray]],
) -> tuple[np.ndarray | None, str | None, int]:
    """Build final NLF vector from ``(tensor_name, weights)`` pairs."""
    if not named_norms:
        return None, None, 0

    norm_weights = [
        w for _, w in sorted(named_norms, key=lambda nw: canonical_tensor_key(nw[0]))
    ]
    sizes = {w.shape[0] for w in norm_weights}
    if len(sizes) == 1:
        return np.concatenate(norm_weights), "direct", len(norm_weights)

    stats_array = np.array(
        [[w.mean(), w.std(), w.max(), w.min()] for w in norm_weights],
        dtype=np.float64,
    ).flatten()
    return stats_array, "stats", len(norm_weights)


def _extract_lep_wsp_wvc(
    state_dict: dict[str, Any],
    light: bool,
    wvc_subsample: int,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[int, np.ndarray] | None]:
    """Single pass over state_dict for LEP, WSP, and WVC."""
    layer_energy: dict[int, float] = defaultdict(float)
    layer_signs: dict[int, dict[str, int]] = defaultdict(
        lambda: {"positive": 0, "near_zero": 0, "total": 0},
    )
    wvc_layer_data: dict[int, list[tuple[str, np.ndarray]]] = defaultdict(list)

    for name, param in state_dict.items():
        layer_idx = extract_layer_index(name)
        if layer_idx is None:
            continue

        n_elem = param.numel()
        if param.dim() < 2 and n_elem < 1024:
            continue

        energy, n_pos, n_nz, n_counted = _tensor_lep_wsp_stats(param, light)

        if param.dim() >= 2:
            layer_energy[layer_idx] += energy

            cat = classify_tensor_name(name)
            if cat not in ("norm", "embedding", "lm_head", "other"):
                wvc_layer_data[layer_idx].append(
                    (name, _wvc_sample(param, wvc_subsample))
                )

        if n_elem >= 1024:
            layer_signs[layer_idx]["positive"] += n_pos
            layer_signs[layer_idx]["near_zero"] += n_nz
            layer_signs[layer_idx]["total"] += n_counted

    return _build_lep(layer_energy), _build_wsp(layer_signs), _build_wvc(wvc_layer_data)


def _tensor_lep_wsp_stats(
    tensor: Any,
    light: bool = False,
) -> tuple[float, int, int, int]:
    """Compute LEP energy + WSP sign counts from a single tensor."""
    n_elem: int = tensor.numel()

    if not tensor.is_floating_point():
        tensor = tensor.float()

    if light and n_elem > _LIGHT_SAMPLE_SIZE:
        flat = tensor.flatten()
        stride = n_elem // _LIGHT_SAMPLE_SIZE
        sample = flat[::stride]
        n_sample = sample.numel()

        n_pos = int((sample > 0).sum())
        n_nz = int((sample.abs() < 1e-4).sum())

        sample_norm_sq = float((sample.float() * sample.float()).sum())
        energy = (sample_norm_sq * n_elem / n_sample) ** 0.5

        return energy, n_pos, n_nz, n_sample

    flat = tensor.flatten()
    n_pos = int((flat > 0).sum())
    n_nz = int((flat.abs() < 1e-4).sum())
    energy = float(torch.norm(flat.float())) if tensor.dim() >= 2 else 0.0
    return energy, n_pos, n_nz, n_elem


def _process_tensor_full(
    tensor: Any,
    tname: str,
    layer_idx: int,
    light: bool,
    wvc_subsample: int,
    layer_energy: dict[int, float],
    layer_signs: dict[int, dict[str, int]],
    wvc_layer_data: dict[int, list[tuple[str, np.ndarray]]],
) -> None:
    """Process a single tensor in the streaming non-sliced path."""
    n_elem = tensor.numel()
    if tensor.dim() < 2 and n_elem < 1024:
        return

    energy, n_pos, n_nz, n_counted = _tensor_lep_wsp_stats(tensor, light)

    if tensor.dim() >= 2:
        layer_energy[layer_idx] += energy

        cat = classify_tensor_name(tname)
        if cat not in ("norm", "embedding", "lm_head", "other"):
            wvc_layer_data[layer_idx].append(
                (tname, _wvc_sample(tensor, wvc_subsample))
            )

    if n_elem >= 1024:
        layer_signs[layer_idx]["positive"] += n_pos
        layer_signs[layer_idx]["near_zero"] += n_nz
        layer_signs[layer_idx]["total"] += n_counted


def _process_tensor_sliced(
    f: Any,
    tname: str,
    layer_idx: int,
    slice_sample_rows: int,
    wvc_subsample: int,
    layer_energy: dict[int, float],
    layer_signs: dict[int, dict[str, int]],
    wvc_layer_data: dict[int, list[tuple[str, np.ndarray]]],
) -> None:
    """Process a tensor via row-slicing for very large models."""
    sl = f.get_slice(tname)
    shape = sl.get_shape()
    ndim = len(shape)
    n_elem = 1
    for s in shape:
        n_elem *= s

    if ndim < 2 and n_elem < 1024:
        return

    if ndim >= 2:
        total_rows = shape[0]
        sample_rows = min(slice_sample_rows, total_rows)
        if sample_rows < total_rows:
            step = total_rows // sample_rows
            indices = list(range(0, total_rows, step))[:sample_rows]
            chunks = [sl[idx : idx + 1, :] for idx in indices]
            tensor = torch.cat(chunks, dim=0).float()
        else:
            tensor = sl[:, :].float()
        scale_factor = total_rows / sample_rows

        energy = (float(torch.sum(tensor**2)) * scale_factor) ** 0.5
        layer_energy[layer_idx] += energy

        flat_t = tensor.flatten()
        n_counted = flat_t.numel()
        n_pos = int((flat_t > 0).sum())
        n_nz = int((flat_t.abs() < 1e-4).sum())
        layer_signs[layer_idx]["positive"] += int(n_pos * scale_factor)
        layer_signs[layer_idx]["near_zero"] += int(n_nz * scale_factor)
        layer_signs[layer_idx]["total"] += int(n_counted * scale_factor)

        cat = classify_tensor_name(tname)
        if cat not in ("norm", "embedding", "lm_head", "other"):
            wvc_layer_data[layer_idx].append(
                (tname, _wvc_sample_from_slice(sl, shape, wvc_subsample))
            )

        del tensor


def _wvc_sample_indices(n_elem: int, wvc_subsample: int) -> np.ndarray | None:
    """Flat element positions sampled for WVC, or ``None`` to keep all.

    Every load path must sample these exact positions so the same weights
    give the same WVC signature.
    """
    if n_elem <= wvc_subsample:
        return None
    stride = n_elem // wvc_subsample
    return np.arange(0, n_elem, stride, dtype=np.int64)[:wvc_subsample]


def _wvc_sample(tensor: Any, wvc_subsample: int) -> np.ndarray:
    """WVC sample from an in-memory tensor."""
    flat = tensor.detach().float().flatten()
    idx = _wvc_sample_indices(flat.numel(), wvc_subsample)
    if idx is not None:
        flat = flat[torch.from_numpy(idx)]
    return flat.cpu().numpy()


def _wvc_sample_from_slice(
    sl: Any, shape: Sequence[int], wvc_subsample: int
) -> np.ndarray:
    """WVC sample read row-by-row from a safetensors slice.

    Reads only the rows (and column span) that contain sampled positions,
    giving the same values as ``_wvc_sample`` on the fully loaded tensor.
    """
    total_rows = int(shape[0])
    row_len = 1
    for s in shape[1:]:
        row_len *= int(s)
    idx = _wvc_sample_indices(total_rows * row_len, wvc_subsample)
    if idx is None:
        return sl[:].float().flatten().cpu().numpy()

    rows, cols = np.divmod(idx, row_len)
    pieces: list[torch.Tensor] = []
    for row in np.unique(rows):
        row_cols = cols[rows == row]
        r = int(row)
        if len(shape) == 2:
            lo, hi = int(row_cols.min()), int(row_cols.max()) + 1
            segment = sl[r : r + 1, lo:hi].flatten()
            pieces.append(segment[torch.from_numpy(row_cols - lo)])
        else:
            segment = sl[r : r + 1].flatten()
            pieces.append(segment[torch.from_numpy(row_cols)])
    return torch.cat(pieces).float().cpu().numpy()


# ── Signal aggregation builders ────────────────────────────────────


def _build_lep(
    layer_energy: dict[int, float],
) -> np.ndarray | None:
    """Build normalised LEP profile from per-layer energy."""
    if not layer_energy:
        return None
    max_layer = max(layer_energy.keys())
    energy = np.array([layer_energy.get(i, 0.0) for i in range(max_layer + 1)])
    max_e = energy.max()
    if max_e > 1e-8:
        energy = energy / max_e
    return energy


def _build_wsp(
    layer_signs: dict[int, dict[str, int]],
) -> np.ndarray | None:
    """Build WSP signature from per-layer sign counts."""
    if not layer_signs:
        return None
    max_layer = max(layer_signs.keys())
    sig: list[float] = []
    for i in range(max_layer + 1):
        s = layer_signs.get(i, {"positive": 0, "near_zero": 0, "total": 1})
        total = max(s["total"], 1)
        sig.extend([s["positive"] / total, s["near_zero"] / total])
    return np.array(sig)


def _build_wvc(
    wvc_layer_data: dict[int, list[tuple[str, np.ndarray]]],
) -> dict[int, np.ndarray] | None:
    """Build WVC per-layer signatures from ``(tensor_name, sample)`` pairs."""
    if not wvc_layer_data:
        return None
    return {
        li: np.concatenate(
            [
                sample
                for _, sample in sorted(
                    wvc_layer_data[li], key=lambda ns: canonical_tensor_key(ns[0])
                )
            ]
        )
        for li in sorted(wvc_layer_data.keys())
    }

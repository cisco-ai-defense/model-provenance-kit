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

"""HuggingFace Hub model loading service.

Strategy decider and full-load executor. Determines whether a model
should be fully loaded into memory or streamed tensor-by-tensor, then
returns a :class:`LoadResult` indicating the outcome.

Loading cascade
---------------
1. Estimate parameter count from ``AutoConfig``.
2. If > ``huge_model_params`` → return ``strategy="streaming"``.
3. Try safetensors single-shard direct load (fast, no model init).
4. Try safetensors multi-shard load (with disk-size guard).
5. Fall back to ``AutoModelForCausalLM`` → ``AutoModel``.
6. On failure → raise :class:`~provenancekit.exceptions.ModelLoadError`.
"""

import json as _json
import os
from pathlib import Path
from typing import Any

import structlog
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load_file
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from provenancekit.config.settings import Settings
from provenancekit.exceptions import ModelLoadError
from provenancekit.models.parsing import ModelFormat
from provenancekit.models.results import LoadResult, LoadStrategy
from provenancekit.utils.tensor import estimate_param_count

log = structlog.get_logger()

_FULL_LOAD_DISK_LIMIT: float = 20e9
_LOCAL_HF_MARKER = "config.json"


def _safe_shard_path(base_dir: str, shard: str) -> str:
    """Join base_dir and shard, rejecting path traversal attempts."""
    result = Path(os.path.join(base_dir, shard)).resolve()  # noqa: PTH118
    if not result.is_relative_to(Path(base_dir).resolve()):
        raise ModelLoadError(
            f"Shard path escapes model directory: {shard}",
            details={"base_dir": base_dir, "shard": shard},
        )
    return str(result)


def resolve_hf_model_ref(model_ref: str) -> str:
    """Return absolute local path for on-disk HF snapshots, else the trimmed ref.

    A **local snapshot** is an existing directory containing ``config.json``
    (standard Hugging Face layout). Otherwise *model_ref* is treated as a
    Hub repo id and returned unchanged (aside from stripping).
    """
    ref = model_ref.strip()
    p = Path(ref).expanduser()
    try:
        resolved = p.resolve()
    except OSError:
        return ref
    if resolved.is_dir() and (resolved / _LOCAL_HF_MARKER).is_file():
        return str(resolved)
    return ref


def is_local_hf_snapshot(model_ref: str) -> bool:
    """True if *model_ref* points at a local HF-format model directory."""
    ref = model_ref.strip()
    p = Path(ref).expanduser()
    try:
        resolved = p.resolve()
    except OSError:
        return False
    return resolved.is_dir() and (resolved / _LOCAL_HF_MARKER).is_file()


def local_safetensors_weight_map(local_root: str) -> dict[str, str] | None:
    """Build tensor name → shard filename map for a local HF folder.

    Reads ``model.safetensors.index.json`` when present; otherwise a single
    ``model.safetensors`` shard (keys enumerated via ``safe_open``).
    Returns ``None`` when no safetensors layout is found.
    """
    root = Path(local_root)
    index_path = root / "model.safetensors.index.json"
    single_path = root / "model.safetensors"
    if index_path.is_file():
        try:
            with open(index_path, encoding="utf-8") as f:  # noqa: PTH123
                data = _json.load(f)
        except (OSError, ValueError, TypeError, KeyError):
            return None
        wm = data.get("weight_map")
        if not isinstance(wm, dict):
            return None
        return {str(k): str(v) for k, v in wm.items()}
    if single_path.is_file():
        try:
            with safe_open(str(single_path), framework="pt") as f:  # type: ignore[no-untyped-call]
                keys = list(f.keys())
        except (OSError, ValueError, RuntimeError) as exc:
            log.debug(
                "local_safetensors_read_failed", path=str(single_path), error=str(exc)
            )
            return None
        return {k: "model.safetensors" for k in keys}
    return None


def local_pytorch_weight_map(local_root: str) -> dict[str, str] | None:
    """Build tensor name -> shard filename map for a local PyTorch HF folder.

    Reads ``pytorch_model.bin.index.json`` when present; otherwise loads
    keys from a single ``pytorch_model.bin`` via ``torch.load`` with
    ``weights_only=True``.
    Returns ``None`` when no PyTorch layout is found.
    """
    root = Path(local_root)
    index_path = root / "pytorch_model.bin.index.json"
    single_path = root / "pytorch_model.bin"
    if index_path.is_file():
        try:
            with open(index_path, encoding="utf-8") as f:  # noqa: PTH123
                data = _json.load(f)
        except (OSError, ValueError, TypeError, KeyError):
            return None
        wm = data.get("weight_map")
        if not isinstance(wm, dict):
            return None
        return {str(k): str(v) for k, v in wm.items()}
    if single_path.is_file():
        try:
            sd = torch.load(
                str(single_path),
                map_location="cpu",
                weights_only=True,
            )
            keys = list(sd.keys())
            del sd
        except (OSError, ValueError, RuntimeError) as exc:
            log.debug(
                "local_pytorch_read_failed", path=str(single_path), error=str(exc)
            )
            return None
        return {k: "pytorch_model.bin" for k in keys}
    return None


def try_hf_download(model_name: str, filename: str) -> str | None:
    """Resolve a Hub file with cache-first probe before network fetch.

    Shared utility used by streaming extraction and format detection.
    Returns the local file path on success, ``None`` on failure.
    """
    for local_only in (True, False):
        try:
            log.info(
                "hf_download_probe",
                model_id=model_name,
                file=filename,
                local_only=local_only,
            )
            kwargs: dict[str, Any] = {"local_files_only": local_only}
            if not local_only:
                kwargs["etag_timeout"] = 10
            return hf_hub_download(model_name, filename, **kwargs)  # type: ignore[no-any-return]
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "hf_download_probe_failed",
                model_id=model_name,
                file=filename,
                local_only=local_only,
                error=str(exc),
            )
            continue
    return None


def detect_model_format(model_name: str) -> ModelFormat | None:
    """Detect the weight file format available for a model.

    Checks for safetensors first (preferred), then PyTorch.
    Works for both local HF snapshots and Hub-cached models.
    Returns ``None`` when no recognisable weight format is found.
    """
    model_name = resolve_hf_model_ref(model_name)

    if is_local_hf_snapshot(model_name):
        root = Path(model_name)
        if (root / "model.safetensors.index.json").is_file():
            return ModelFormat.SAFETENSORS
        if (root / "model.safetensors").is_file():
            return ModelFormat.SAFETENSORS
        if (root / "pytorch_model.bin.index.json").is_file():
            return ModelFormat.PYTORCH
        if (root / "pytorch_model.bin").is_file():
            return ModelFormat.PYTORCH
        return None

    for filename, fmt in [
        ("model.safetensors.index.json", ModelFormat.SAFETENSORS),
        ("model.safetensors", ModelFormat.SAFETENSORS),
        ("pytorch_model.bin.index.json", ModelFormat.PYTORCH),
        ("pytorch_model.bin", ModelFormat.PYTORCH),
    ]:
        if try_hf_download(model_name, filename) is not None:
            return fmt
    return None


# ── Public API ─────────────────────────────────────────────────────


def estimate_model_params(
    model_name: str,
    *,
    trust_remote_code: bool = False,
) -> int:
    """Estimate parameter count from HuggingFace config.

    Loads ``AutoConfig`` and delegates to the shared
    :func:`~provenancekit.utils.tensor.estimate_param_count` heuristic
    which accounts for GQA/MQA, gated MLPs, and tied embeddings.
    Returns ``0`` on any failure.
    """
    try:
        config: Any = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        return estimate_param_count(config)
    except Exception as exc:  # noqa: BLE001
        log.debug("param_count_estimate_failed", model_id=model_name, error=str(exc))
        return 0


def load_state_dict(
    model_name: str,
    settings: Settings | None = None,
) -> LoadResult:
    """Load model weights via safetensors or AutoModel fallback.

    Returns a :class:`LoadResult` whose ``strategy`` field tells the
    caller what to do next:

    * ``full`` — ``state_dict`` is populated; use it directly.
    * ``streaming`` — model too large for full load; stream tensors.

    Args:
        model_name: HuggingFace model identifier or local path.
        settings: Runtime configuration.  Defaults to ``Settings()``.

    Raises:
        ModelLoadError: When the model cannot be loaded by any strategy.
    """
    if settings is None:
        settings = Settings()
    model_name = resolve_hf_model_ref(model_name)

    log.info("estimating_params", model_id=model_name)
    est = estimate_model_params(
        model_name, trust_remote_code=settings.trust_remote_code
    )
    log.info("param_estimate", model_id=model_name, params=est)
    if est > settings.huge_model_params:
        return LoadResult(
            strategy=LoadStrategy.streaming,
            source="too_large_for_full_load",
        )

    # Hybrid local-size guard:
    # 1) prefer precise shard/index-derived weight bytes (safetensors + pytorch)
    # 2) fallback to full local directory size when index data is unavailable
    if is_local_hf_snapshot(model_name):
        local_bytes, local_method = _estimate_local_size_for_streaming(model_name)
        if local_bytes is not None:
            log.info(
                "local_size_estimated",
                model_id=model_name,
                method=local_method,
                bytes_on_disk=local_bytes,
                limit=_FULL_LOAD_DISK_LIMIT,
            )
            if local_bytes > _FULL_LOAD_DISK_LIMIT:
                log.info(
                    "streaming_selected_by_local_size",
                    model_id=model_name,
                    method=local_method,
                    bytes_on_disk=local_bytes,
                    limit=_FULL_LOAD_DISK_LIMIT,
                )
                return LoadResult(
                    strategy=LoadStrategy.streaming,
                    source="too_large_for_full_load",
                )

    try:
        config: Any = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=settings.trust_remote_code,
        )
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load config for '{model_name}': {exc}",
            details={"stage": "config_load"},
            model_id=model_name,
        ) from exc

    log.info("trying_safetensors", model_id=model_name)
    result = _try_safetensors(model_name, config)
    if result is not None:
        log.info("safetensors_loaded", model_id=model_name, source=result.source)
        return result

    log.info("trying_automodel", model_id=model_name)
    return _try_automodel(
        model_name, config, trust_remote_code=settings.trust_remote_code
    )


# ── Private helpers ────────────────────────────────────────────────


def _try_safetensors(
    model_name: str,
    config: Any,
) -> LoadResult | None:
    """Attempt safetensors loading (single-shard then multi-shard)."""
    result = _try_safetensors_single(
        model_name,
        config,
        hf_hub_download,
        safetensors_load_file,
    )
    if result is not None:
        return result

    return _try_safetensors_sharded(
        model_name,
        config,
        hf_hub_download,
        safetensors_load_file,
    )


def _try_safetensors_single(
    model_name: str,
    config: Any,
    hf_hub_download: Any,
    load_file: Any,
) -> LoadResult | None:
    """Try loading a single-shard safetensors model."""
    if is_local_hf_snapshot(model_name):
        path = os.path.join(model_name, "model.safetensors")  # noqa: PTH118
        if os.path.isfile(path):  # noqa: PTH113
            try:
                log.info(
                    "safetensors_single_ready",
                    model_id=model_name,
                    local_only=True,
                    file="model.safetensors",
                )
                state: dict[str, Any] = load_file(path, device="cpu")
                return LoadResult(
                    state_dict=state,
                    config=config,
                    strategy=LoadStrategy.full,
                    source="safetensors",
                )
            except (OSError, ValueError, RuntimeError) as exc:
                log.debug(
                    "safetensors_single_local_load_failed",
                    model_id=model_name,
                    error=str(exc),
                )
                return None
        return None

    for local_only in (True, False):
        try:
            log.info(
                "safetensors_single_probe",
                model_id=model_name,
                local_only=local_only,
            )
            path = hf_hub_download(
                model_name,
                "model.safetensors",
                local_files_only=local_only,
            )
            log.info(
                "safetensors_single_ready",
                model_id=model_name,
                local_only=local_only,
                file="model.safetensors",
            )
            state = load_file(path, device="cpu")
            return LoadResult(
                state_dict=state,
                config=config,
                strategy=LoadStrategy.full,
                source="safetensors",
            )
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "safetensors_single_probe_failed",
                model_id=model_name,
                local_only=local_only,
                error=str(exc),
            )
    return None


def _try_safetensors_sharded(
    model_name: str,
    config: Any,
    hf_hub_download: Any,
    load_file: Any,
) -> LoadResult | None:
    """Try loading a multi-shard safetensors model."""
    if is_local_hf_snapshot(model_name):
        idx_path = os.path.join(  # noqa: PTH118
            model_name,
            "model.safetensors.index.json",
        )
        if not os.path.isfile(idx_path):  # noqa: PTH113
            return None
        try:
            with open(idx_path, encoding="utf-8") as f:  # noqa: PTH123
                idx = _json.load(f)
            shard_files = sorted(set(idx["weight_map"].values()))
            base_dir = model_name

            total_bytes = 0
            for shard in shard_files:
                sp = _safe_shard_path(base_dir, shard)
                if os.path.isfile(sp):  # noqa: PTH113
                    total_bytes += os.path.getsize(sp)  # noqa: PTH202
            log.info(
                "safetensors_shards_indexed",
                model_id=model_name,
                n_shards=len(shard_files),
                local_only=True,
            )
            if total_bytes > _FULL_LOAD_DISK_LIMIT:
                log.info(
                    "streaming_selected_by_disk_limit",
                    model_id=model_name,
                    bytes_on_disk=total_bytes,
                    limit=_FULL_LOAD_DISK_LIMIT,
                )
                return LoadResult(
                    config=config,
                    strategy=LoadStrategy.streaming,
                    source="too_large_for_full_load",
                )

            state: dict[str, Any] = {}
            total = len(shard_files)
            for i, shard in enumerate(shard_files, 1):
                shard_path = _safe_shard_path(base_dir, shard)
                if not os.path.isfile(shard_path):  # noqa: PTH113
                    return None
                log.info(
                    "using_cached_shard",
                    model_id=model_name,
                    shard=shard,
                    shard_progress=f"{i}/{total}",
                )
                state.update(load_file(shard_path, device="cpu"))
            return LoadResult(
                state_dict=state,
                config=config,
                strategy=LoadStrategy.full,
                source="safetensors_sharded",
            )
        except (OSError, ValueError, RuntimeError) as exc:
            log.debug(
                "safetensors_sharded_local_load_failed",
                model_id=model_name,
                error=str(exc),
            )
            return None

    for local_only in (True, False):
        try:
            log.info(
                "safetensors_index_probe",
                model_id=model_name,
                local_only=local_only,
            )
            idx_path = hf_hub_download(
                model_name,
                "model.safetensors.index.json",
                local_files_only=local_only,
            )
            with open(idx_path, encoding="utf-8") as f:  # noqa: PTH123
                idx = _json.load(f)
            shard_files = sorted(set(idx["weight_map"].values()))
            base_dir = os.path.dirname(idx_path)  # noqa: PTH120

            total_bytes = 0
            for shard in shard_files:
                sp = _safe_shard_path(base_dir, shard)
                if os.path.isfile(sp):  # noqa: PTH113
                    total_bytes += os.path.getsize(sp)  # noqa: PTH202
            log.info(
                "safetensors_shards_indexed",
                model_id=model_name,
                n_shards=len(shard_files),
                local_only=local_only,
            )
            if total_bytes > _FULL_LOAD_DISK_LIMIT:
                log.info(
                    "streaming_selected_by_disk_limit",
                    model_id=model_name,
                    bytes_on_disk=total_bytes,
                    limit=_FULL_LOAD_DISK_LIMIT,
                )
                return LoadResult(
                    config=config,
                    strategy=LoadStrategy.streaming,
                    source="too_large_for_full_load",
                )

            state = {}
            total = len(shard_files)
            for i, shard in enumerate(shard_files, 1):
                shard_path = _safe_shard_path(base_dir, shard)
                if not os.path.exists(shard_path):  # noqa: PTH110
                    log.info(
                        "downloading_shard",
                        model_id=model_name,
                        shard=shard,
                        shard_progress=f"{i}/{total}",
                    )
                    shard_path = hf_hub_download(
                        model_name,
                        shard,
                        local_files_only=local_only,
                    )
                else:
                    log.info(
                        "using_cached_shard",
                        model_id=model_name,
                        shard=shard,
                        shard_progress=f"{i}/{total}",
                    )
                state.update(load_file(shard_path, device="cpu"))
            return LoadResult(
                state_dict=state,
                config=config,
                strategy=LoadStrategy.full,
                source="safetensors_sharded",
            )
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "safetensors_sharded_probe_failed",
                model_id=model_name,
                local_only=local_only,
                error=str(exc),
            )
    return None


def _try_automodel(
    model_name: str,
    config: Any,
    *,
    trust_remote_code: bool = False,
) -> LoadResult:
    """Fall back to AutoModel loading (slow but universal).

    Raises:
        ModelLoadError: When neither AutoModelForCausalLM nor AutoModel
            can load the model.
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        state: dict[str, Any] = model.state_dict()
        cfg = model.config
        del model
        return LoadResult(
            state_dict=state,
            config=cfg,
            strategy=LoadStrategy.full,
            source="automodel_causal",
        )
    except (ValueError, OSError, RuntimeError):
        try:
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )
            state = model.state_dict()
            cfg = model.config
            del model
            return LoadResult(
                state_dict=state,
                config=cfg,
                strategy=LoadStrategy.full,
                source="automodel_auto",
            )
        except Exception as exc:
            raise ModelLoadError(
                f"All AutoModel strategies failed for '{model_name}': {exc}",
                details={"stage": "automodel", "tried": ["causal", "auto"]},
                model_id=model_name,
            ) from exc
    except Exception as exc:
        raise ModelLoadError(
            f"Model loading failed for '{model_name}': {exc}",
            details={"stage": "automodel_causal"},
            model_id=model_name,
        ) from exc


def _estimate_local_size_for_streaming(local_root: str) -> tuple[int | None, str]:
    """Estimate local model size for streaming decision.

    Returns:
        (bytes, method) where method is one of:
        - safetensors_index
        - pytorch_index
        - safetensors_single
        - pytorch_single
        - directory_fallback
        - unavailable
    """
    root = Path(local_root)

    safetensors_index = _sum_index_shards(
        root, index_name="model.safetensors.index.json"
    )
    if safetensors_index is not None:
        return safetensors_index, "safetensors_index"

    pytorch_index = _sum_index_shards(root, index_name="pytorch_model.bin.index.json")
    if pytorch_index is not None:
        return pytorch_index, "pytorch_index"

    safetensors_single = root / "model.safetensors"
    if safetensors_single.is_file():
        try:
            return safetensors_single.stat().st_size, "safetensors_single"
        except OSError:
            pass

    pytorch_single = root / "pytorch_model.bin"
    if pytorch_single.is_file():
        try:
            return pytorch_single.stat().st_size, "pytorch_single"
        except OSError:
            pass

    directory_total = _sum_directory_bytes(root)
    if directory_total is not None:
        return directory_total, "directory_fallback"
    return None, "unavailable"


def _sum_index_shards(root: Path, *, index_name: str) -> int | None:
    """Sum bytes for shards listed in a Hugging Face index file."""
    index_path = root / index_name
    if not index_path.is_file():
        return None
    try:
        data = _json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None

    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        return None

    shard_files = {str(v) for v in weight_map.values()}
    total = 0
    any_found = False
    for shard in shard_files:
        shard_path = root / shard
        try:
            if shard_path.is_file():
                total += shard_path.stat().st_size
                any_found = True
        except OSError:
            continue
    return total if any_found else None


def _sum_directory_bytes(root: Path) -> int | None:
    """Best-effort recursive size sum for a local model directory."""
    total = 0
    any_file = False
    try:
        for entry in root.rglob("*"):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
                    any_file = True
            except OSError:
                continue
    except OSError:
        return None
    return total if any_file else None

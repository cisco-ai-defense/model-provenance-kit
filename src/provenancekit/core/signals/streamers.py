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

"""Format-agnostic tensor streaming backends.

Provides a :class:`TensorStreamer` protocol and concrete implementations
for safetensors and PyTorch weight files.  Each streamer knows how to:

* Build a ``weight_map`` (tensor name -> shard filename).
* Open individual shard files for tensor-by-tensor reading.
* Load a single tensor by name from an open shard.

The streaming extraction loop in :mod:`weight_signals` delegates all
format-specific I/O to these streamers so new formats can be added by
implementing the same protocol.
"""

import json as _json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

import structlog
import torch
from safetensors import safe_open

from provenancekit.exceptions import ExtractionError
from provenancekit.models.parsing import ModelFormat
from provenancekit.services.model_loader import (
    is_local_hf_snapshot,
    local_pytorch_weight_map,
    local_safetensors_weight_map,
    resolve_hf_model_ref,
    try_hf_download,
)

log = structlog.get_logger()


# ── Protocol ──────────────────────────────────────────────────────


@runtime_checkable
class TensorStreamer(Protocol):
    """Minimal interface for format-specific tensor streaming."""

    @property
    def format(self) -> ModelFormat: ...  # noqa: D102

    @property
    def weight_map(self) -> dict[str, str]:
        """tensor_name -> shard_filename."""
        ...

    @contextmanager
    def open_shard(self, shard_path: str) -> Iterator[Any]:
        """Context manager yielding a handle for reading tensors."""
        ...

    def get_tensor(self, handle: Any, tensor_name: str) -> torch.Tensor:
        """Load a single tensor from an open shard handle."""
        ...

    def get_slice(self, handle: Any, tensor_name: str) -> Any:
        """Return a slice object for row-level access.

        Implementations that do not support slicing should raise
        ``NotImplementedError``.
        """
        ...

    @property
    def supports_slicing(self) -> bool:
        """Whether this streamer supports ``get_slice``."""
        ...


# ── Safetensors ───────────────────────────────────────────────────


class SafetensorsStreamer:
    """Stream tensors from ``.safetensors`` shard files."""

    def __init__(self, wmap: dict[str, str]) -> None:  # noqa: D107
        self._weight_map = wmap

    @property
    def format(self) -> ModelFormat:  # noqa: D102
        return ModelFormat.SAFETENSORS

    @property
    def weight_map(self) -> dict[str, str]:  # noqa: D102
        return self._weight_map

    @contextmanager
    def open_shard(self, shard_path: str) -> Iterator[Any]:  # noqa: D102
        with safe_open(shard_path, framework="pt", device="cpu") as f:  # type: ignore[no-untyped-call]
            yield f

    def get_tensor(self, handle: Any, tensor_name: str) -> torch.Tensor:  # noqa: D102
        return handle.get_tensor(tensor_name)  # type: ignore[no-any-return]

    def get_slice(self, handle: Any, tensor_name: str) -> Any:  # noqa: D102
        return handle.get_slice(tensor_name)

    @property
    def supports_slicing(self) -> bool:  # noqa: D102
        return True


# ── PyTorch ───────────────────────────────────────────────────────


class _PytorchShardHandle:
    """Thin wrapper around a loaded PyTorch shard dict."""

    __slots__ = ("_sd",)

    def __init__(self, sd: dict[str, Any]) -> None:
        self._sd = sd

    def get_tensor(self, tensor_name: str) -> torch.Tensor:
        return self._sd[tensor_name]  # type: ignore[no-any-return]


class PytorchStreamer:
    """Stream tensors from ``.bin`` PyTorch shard files.

    Uses ``torch.load(mmap=True, weights_only=True)`` when available
    (PyTorch >= 2.1) to memory-map shards, keeping peak RAM low.
    """

    def __init__(self, wmap: dict[str, str]) -> None:  # noqa: D107
        self._weight_map = wmap
        major, minor = (int(x) for x in torch.__version__.split(".")[:2])
        self._supports_mmap = (major, minor) >= (2, 1)

    @property
    def format(self) -> ModelFormat:  # noqa: D102
        return ModelFormat.PYTORCH

    @property
    def weight_map(self) -> dict[str, str]:  # noqa: D102
        return self._weight_map

    @contextmanager
    def open_shard(self, shard_path: str) -> Iterator[_PytorchShardHandle]:  # noqa: D102
        kwargs: dict[str, Any] = {
            "map_location": "cpu",
            "weights_only": True,
        }
        if self._supports_mmap:
            kwargs["mmap"] = True
        sd = torch.load(shard_path, **kwargs)  # noqa: S614
        try:
            yield _PytorchShardHandle(sd)
        finally:
            del sd

    def get_tensor(self, handle: Any, tensor_name: str) -> torch.Tensor:  # noqa: D102
        return handle.get_tensor(tensor_name)  # type: ignore[no-any-return]

    def get_slice(self, handle: Any, tensor_name: str) -> Any:  # noqa: D102
        raise NotImplementedError("PyTorch format does not support row-slicing")

    @property
    def supports_slicing(self) -> bool:  # noqa: D102
        return False


# ── Weight-map resolution helpers ─────────────────────────────────


def _resolve_safetensors_weight_map(
    model_name: str,
) -> dict[str, str] | None:
    """Build a safetensors weight map for local or Hub models."""
    if is_local_hf_snapshot(model_name):
        return local_safetensors_weight_map(model_name)

    idx_path = try_hf_download(model_name, "model.safetensors.index.json")
    if idx_path is not None:
        try:
            with open(idx_path, encoding="utf-8") as f:  # noqa: PTH123
                idx = _json.load(f)
            wm = idx.get("weight_map", {})
            if isinstance(wm, dict) and wm:
                return {str(k): str(v) for k, v in wm.items()}
        except (OSError, ValueError, TypeError, KeyError):
            pass

    single_path = try_hf_download(model_name, "model.safetensors")
    if single_path is not None:
        try:
            with safe_open(single_path, framework="pt", device="cpu") as f:  # type: ignore[no-untyped-call]
                names = list(f.keys())
            if names:
                return {n: "model.safetensors" for n in names}
        except (OSError, ValueError, RuntimeError) as exc:
            log.debug("safetensors_single_read_failed", error=str(exc))

    return None


def _resolve_pytorch_weight_map(
    model_name: str,
) -> dict[str, str] | None:
    """Build a PyTorch weight map for local or Hub models."""
    if is_local_hf_snapshot(model_name):
        return local_pytorch_weight_map(model_name)

    idx_path = try_hf_download(model_name, "pytorch_model.bin.index.json")
    if idx_path is not None:
        try:
            with open(idx_path, encoding="utf-8") as f:  # noqa: PTH123
                idx = _json.load(f)
            wm = idx.get("weight_map", {})
            if isinstance(wm, dict) and wm:
                return {str(k): str(v) for k, v in wm.items()}
        except (OSError, ValueError, TypeError, KeyError):
            pass

    single_path = try_hf_download(model_name, "pytorch_model.bin")
    if single_path is not None:
        try:
            sd = torch.load(single_path, map_location="cpu", weights_only=True)
            names = list(sd.keys())
            del sd
            if names:
                return {n: "pytorch_model.bin" for n in names}
        except (OSError, ValueError, RuntimeError) as exc:
            log.debug("pytorch_single_read_failed", error=str(exc))

    return None


# ── Factory ───────────────────────────────────────────────────────


def create_streamer(model_name: str) -> TensorStreamer:
    """Create the appropriate streamer for *model_name*.

    Tries safetensors first (preferred), then PyTorch.
    """
    model_name = resolve_hf_model_ref(model_name)

    wmap = _resolve_safetensors_weight_map(model_name)
    if wmap:
        log.info(
            "streamer_created",
            model_id=model_name,
            format="safetensors",
            n_tensors=len(wmap),
        )
        return SafetensorsStreamer(wmap)

    wmap = _resolve_pytorch_weight_map(model_name)
    if wmap:
        log.info(
            "streamer_created",
            model_id=model_name,
            format="pytorch",
            n_tensors=len(wmap),
        )
        return PytorchStreamer(wmap)

    raise ExtractionError(
        f"No supported weight format (safetensors or pytorch) found for '{model_name}'",
        details={"model_id": model_name, "stage": "streamer_creation"},
    )

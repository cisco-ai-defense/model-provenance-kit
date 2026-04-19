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

"""Tests for core/signals/streamers.py — format-agnostic tensor streaming."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from provenancekit.core.signals.streamers import (
    PytorchStreamer,
    SafetensorsStreamer,
    TensorStreamer,
    create_streamer,
)
from provenancekit.exceptions import ExtractionError
from provenancekit.models.parsing import ModelFormat

# ── SafetensorsStreamer ───────────────────────────────────────────


class TestSafetensorsStreamer:
    def test_implements_protocol(self) -> None:
        assert isinstance(SafetensorsStreamer({}), TensorStreamer)

    def test_format_is_safetensors(self) -> None:
        s = SafetensorsStreamer({"a": "shard.safetensors"})
        assert s.format == ModelFormat.SAFETENSORS

    def test_supports_slicing(self) -> None:
        assert SafetensorsStreamer({}).supports_slicing is True

    def test_open_shard_and_get_tensor(self, tmp_path: Path) -> None:
        t = torch.randn(3, 4)
        path = str(tmp_path / "shard.safetensors")
        save_file({"layer.weight": t}, path)
        streamer = SafetensorsStreamer({"layer.weight": "shard.safetensors"})
        with streamer.open_shard(path) as handle:
            loaded = streamer.get_tensor(handle, "layer.weight")
        assert torch.allclose(loaded, t)

    def test_get_slice(self, tmp_path: Path) -> None:
        t = torch.randn(8, 4)
        path = str(tmp_path / "shard.safetensors")
        save_file({"w": t}, path)
        streamer = SafetensorsStreamer({"w": "shard.safetensors"})
        with streamer.open_shard(path) as handle:
            sl = streamer.get_slice(handle, "w")
            shape = sl.get_shape()
        assert shape == [8, 4]


# ── PytorchStreamer ───────────────────────────────────────────────


class TestPytorchStreamer:
    def test_implements_protocol(self) -> None:
        assert isinstance(PytorchStreamer({}), TensorStreamer)

    def test_format_is_pytorch(self) -> None:
        s = PytorchStreamer({"a": "pytorch_model.bin"})
        assert s.format == ModelFormat.PYTORCH

    def test_does_not_support_slicing(self) -> None:
        assert PytorchStreamer({}).supports_slicing is False

    def test_get_slice_raises(self, tmp_path: Path) -> None:
        sd = {"w": torch.randn(4)}
        path = str(tmp_path / "model.bin")
        torch.save(sd, path)
        streamer = PytorchStreamer({"w": "model.bin"})
        with streamer.open_shard(path) as handle, pytest.raises(NotImplementedError):
            streamer.get_slice(handle, "w")

    def test_open_shard_and_get_tensor(self, tmp_path: Path) -> None:
        t = torch.randn(3, 4)
        sd: dict[str, Any] = {"layer.weight": t}
        path = str(tmp_path / "pytorch_model.bin")
        torch.save(sd, path)
        streamer = PytorchStreamer({"layer.weight": "pytorch_model.bin"})
        with streamer.open_shard(path) as handle:
            loaded = streamer.get_tensor(handle, "layer.weight")
        assert torch.allclose(loaded, t)

    def test_shard_memory_released(self, tmp_path: Path) -> None:
        sd: dict[str, Any] = {"w": torch.randn(100, 100)}
        path = str(tmp_path / "model.bin")
        torch.save(sd, path)
        streamer = PytorchStreamer({"w": "model.bin"})
        with streamer.open_shard(path) as handle:
            _ = streamer.get_tensor(handle, "w")


# ── create_streamer factory ───────────────────────────────────────


class TestCreateStreamer:
    def test_creates_safetensors_streamer_for_safetensors_model(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        save_file({"w": torch.randn(4)}, str(tmp_path / "model.safetensors"))
        streamer = create_streamer(str(tmp_path))
        assert isinstance(streamer, SafetensorsStreamer)
        assert "w" in streamer.weight_map

    def test_creates_pytorch_streamer_for_pytorch_model(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        torch.save({"w": torch.randn(4)}, str(tmp_path / "pytorch_model.bin"))
        streamer = create_streamer(str(tmp_path))
        assert isinstance(streamer, PytorchStreamer)
        assert "w" in streamer.weight_map

    def test_prefers_safetensors_over_pytorch(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        save_file({"st": torch.randn(4)}, str(tmp_path / "model.safetensors"))
        torch.save({"pt": torch.randn(4)}, str(tmp_path / "pytorch_model.bin"))
        streamer = create_streamer(str(tmp_path))
        assert isinstance(streamer, SafetensorsStreamer)
        assert "st" in streamer.weight_map

    def test_raises_when_no_weights_found(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ExtractionError, match="No supported weight format"):
            create_streamer(str(tmp_path))

    def test_pytorch_sharded_from_index(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        idx = {
            "weight_map": {
                "layer.0.weight": "pytorch_model-00001.bin",
                "layer.1.weight": "pytorch_model-00002.bin",
            },
        }
        (tmp_path / "pytorch_model.bin.index.json").write_text(
            json.dumps(idx),
            encoding="utf-8",
        )
        streamer = create_streamer(str(tmp_path))
        assert isinstance(streamer, PytorchStreamer)
        assert len(streamer.weight_map) == 2

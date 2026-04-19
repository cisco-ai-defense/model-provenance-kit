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

from provenancekit.models.parsing import ModelFormat, TensorMetadata


class TestModelFormat:
    def test_enum_values(self):
        assert ModelFormat.SAFETENSORS.value == "safetensors"
        assert ModelFormat.PYTORCH.value == "pytorch"

    def test_string_comparison(self):
        assert ModelFormat.SAFETENSORS == "safetensors"
        assert ModelFormat.PYTORCH == "pytorch"

    def test_membership(self):
        assert "safetensors" in [e.value for e in ModelFormat]


class TestTensorMetadata:
    def test_valid(self):
        tm = TensorMetadata(
            name="model.layers.0.self_attn.q_proj.weight",
            shape=(4096, 4096),
            dtype="float16",
            category="q_proj",
            layer_index=0,
        )
        assert tm.name == "model.layers.0.self_attn.q_proj.weight"
        assert tm.shape == (4096, 4096)
        assert tm.dtype == "float16"
        assert tm.category == "q_proj"
        assert tm.layer_index == 0

    def test_no_layer_index(self):
        tm = TensorMetadata(
            name="model.embed_tokens.weight",
            shape=(32000, 4096),
            dtype="float16",
            category="embedding",
            layer_index=None,
        )
        assert tm.layer_index is None
        assert tm.category == "embedding"

    def test_high_dimensional_shape(self):
        tm = TensorMetadata(
            name="model.layers.5.mlp.gate_proj.weight",
            shape=(11008, 4096),
            dtype="bfloat16",
            category="gate_proj",
            layer_index=5,
        )
        assert tm.shape == (11008, 4096)

    def test_serialization_round_trip(self):
        tm = TensorMetadata(
            name="model.layers.0.self_attn.k_proj.weight",
            shape=(1024, 4096),
            dtype="float16",
            category="k_proj",
            layer_index=0,
        )
        data = tm.model_dump()
        restored = TensorMetadata.model_validate(data)
        assert restored == tm

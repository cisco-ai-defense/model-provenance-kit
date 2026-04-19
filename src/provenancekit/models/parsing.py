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

"""Pydantic models for parsed model file metadata."""

from enum import StrEnum

from pydantic import BaseModel


class ModelFormat(StrEnum):
    """Supported model file formats."""

    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"


class TensorMetadata(BaseModel):
    """Parsed tensor information from a model file.

    ``category`` comes from ``utils.tensor.classify_tensor_name`` and
    ``layer_index`` from ``utils.tensor.extract_layer_index``.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    category: str
    layer_index: int | None

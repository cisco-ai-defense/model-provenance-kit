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

from provenancekit.config.constants import (
    FAMILY_MAP,
    IDENTITY_WEIGHTS,
    TOKENIZER_WEIGHTS,
)
from provenancekit.config.settings import Settings


def test_identity_weights_sum_to_one():
    assert abs(sum(IDENTITY_WEIGHTS.values()) - 1.0) < 1e-6


def test_family_map_has_llama():
    assert "llama" in FAMILY_MAP
    assert "llama" in FAMILY_MAP["llama"]


def test_tokenizer_weights_sum_to_one():
    assert abs(sum(TOKENIZER_WEIGHTS.values()) - 1.0) < 1e-6


def test_huge_model_threshold():
    assert Settings().huge_model_params == 10e9

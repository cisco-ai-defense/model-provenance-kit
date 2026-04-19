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

"""Abstract base classes for provenance scanners.

Defines the contract that all scanner implementations must follow.
Enables future extensibility (e.g., data provenance, code provenance).
"""

from abc import ABC, abstractmethod

from provenancekit.models.results import CompareResult, ScanResult


class BaseScanner(ABC):
    """Abstract interface for provenance scanners."""

    @abstractmethod
    def compare(self, model_a: str, model_b: str) -> CompareResult:
        """Compare two models and return a detailed provenance result."""

    @abstractmethod
    def scan(
        self,
        model_id: str,
        *,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> ScanResult:
        """Scan a model against a reference database.

        Args:
            model_id: HuggingFace model identifier or local path.
            top_k: Maximum number of matches to return.
            threshold: Minimum pipeline score for inclusion.
        """

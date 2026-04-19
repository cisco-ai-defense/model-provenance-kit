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

"""Custom exception hierarchy for ProvenanceKit.

All custom exceptions inherit from :class:`ProvenanceError`, allowing
callers to catch every provenance-specific error with a single clause
or to handle individual subtypes with precision.

Example::

    from provenancekit.exceptions import ModelLoadError, ProvenanceError

    try:
        result = scanner.compare("model_a", "model_b")
    except ModelLoadError as exc:
        print(f"Model load problem ({exc.model_id}): {exc}")
        print(f"Details: {exc.details}")
    except ProvenanceError as exc:
        print(f"Provenance error: {exc}")
"""

from typing import Any


class ProvenanceError(Exception):
    """Base exception for all ProvenanceKit errors.

    All custom exceptions in the toolkit inherit from this base class,
    allowing callers to catch all provenance-specific errors with a
    single ``except`` clause if desired.

    Args:
        message: Human-readable error description.
        details: Optional dict with additional machine-readable context.
    """

    def __init__(  # noqa: D107
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the error for structured logging or API responses."""
        return {
            "error_type": type(self).__name__,
            "message": self.message,
            "details": self.details,
        }


class ExtractionError(ProvenanceError):
    """Raised when signal or fingerprint extraction fails.

    This covers failures in MFI fingerprinting, tokenizer feature
    extraction, and weight-signal extraction (both full and streaming).

    Example::

        raise ExtractionError(
            "No safetensors weight map found",
            {"model_id": "org/model", "stage": "streaming_setup"},
        )
    """


class ModelLoadError(ExtractionError):
    """Raised when a model cannot be loaded from Hub or local path.

    This is a subtype of :class:`ExtractionError` because a load
    failure prevents extraction. Carries the model identifier for
    programmatic handling.

    Example::

        raise ModelLoadError(
            "Config download failed",
            {"model_id": "org/model", "strategy": "automodel"},
        )
    """

    def __init__(  # noqa: D107
        self,
        message: str,
        details: dict[str, Any] | None = None,
        *,
        model_id: str = "",
    ) -> None:
        super().__init__(message, details)
        self.model_id = model_id


class CacheError(ProvenanceError):
    """Raised on fingerprint/result cache read or write failures.

    Cache errors are **non-fatal** by design — the ``CacheService``
    catches them internally so a broken cache never aborts a comparison.

    Example::

        raise CacheError(
            "Corrupt cache file",
            {"path": "/home/user/.provenancekit/cache/org__model.json"},
        )
    """

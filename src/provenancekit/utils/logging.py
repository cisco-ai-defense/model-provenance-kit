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

"""Structured logging setup (structlog).

Centralises logging configuration so both the CLI and any future
programmatic entry points share a single setup path.
"""

import logging
import sys

import structlog

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "filelock",
    "huggingface_hub",
)

_configured = False


def configure_logging(verbose: bool) -> None:
    """Set up structlog to print human-readable lines to stderr.

    Safe to call multiple times — subsequent calls are no-ops unless
    the module-level ``_configured`` flag is reset (useful for testing).

    Args:
        verbose: When ``True`` emit INFO-level structured logs;
            otherwise only WARNING and above.
    """
    global _configured  # noqa: PLW0603
    if _configured:
        return
    _configured = True

    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level,
    )
    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(
            file=sys.stderr,
        ),
    )

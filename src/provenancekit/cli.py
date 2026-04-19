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

"""CLI entry point for provenancekit.

Usage::

    provenancekit compare MODEL_A MODEL_B [options]
    provenancekit scan MODEL_ID [options]
    provenancekit download-deepsignals-fingerprint [options]
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

_import_start = time.monotonic()
sys.stderr.write(
    "🔍 provenancekit: loading modules (first run may take a few seconds)…"
)
sys.stderr.flush()

from rich.console import Console  # noqa: E402

from provenancekit.config.settings import Settings  # noqa: E402
from provenancekit.core.results.formatters import (  # noqa: E402
    format_json,
    format_plain,
    format_scan_json,
    format_scan_plain,
    format_scan_table,
    format_table,
)
from provenancekit.exceptions import ProvenanceError  # noqa: E402
from provenancekit.services.cache import CacheService, NullCache  # noqa: E402
from provenancekit.services.download import (  # noqa: E402
    download_deep_signals,
    has_deep_signals,
    show_deep_signals_status,
)
from provenancekit.utils.logging import configure_logging  # noqa: E402

_import_ms = (time.monotonic() - _import_start) * 1000
sys.stderr.write(f"\r\033[2K🔍 provenancekit: ready ({_import_ms:.0f}ms)\n")
sys.stderr.flush()


def _positive_int(value: str) -> int:
    """Argparse type: positive integer (>= 1)."""
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return ivalue


def _unit_float(value: str) -> float:
    """Argparse type: float in [0.0, 1.0]."""
    fvalue = float(value)
    if not 0.0 <= fvalue <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in [0.0, 1.0], got {value}")
    return fvalue


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="provenancekit",
        description="Model provenance detection toolkit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose structured logging to stderr",
    )
    sub = parser.add_subparsers(dest="command")

    cmp = sub.add_parser("compare", help="Compare two models for provenance")
    cmp.add_argument(
        "model_a",
        help="First model: HF repo id (e.g. gpt2) or local path",
    )
    cmp.add_argument(
        "model_b",
        help="Second model: Hub repo id or local snapshot path",
    )
    cmp_fmt = cmp.add_mutually_exclusive_group()
    cmp_fmt.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Output as JSON",
    )
    cmp_fmt.add_argument(
        "--plain",
        dest="output_plain",
        action="store_true",
        help="Output as plain key-value text (no colour)",
    )
    cmp.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override the default cache directory",
    )
    cmp.add_argument(
        "--no-cache",
        dest="no_cache",
        action="store_true",
        help="Disable feature caching",
    )
    cmp.add_argument(
        "--timing",
        dest="show_timing",
        action="store_true",
        help="Show high-level phase timings in table/plain output",
    )
    cmp.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        help="Allow execution of model-hosted Python code (config/tokenizer). "
        "Use only with models you trust.",
    )

    scn = sub.add_parser("scan", help="Scan a model against known models")
    scn.add_argument(
        "model_id",
        help="Model to scan: Hub repo id or path to a local HF snapshot directory",
    )
    scn_fmt = scn.add_mutually_exclusive_group()
    scn_fmt.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Output as JSON",
    )
    scn_fmt.add_argument(
        "--plain",
        dest="output_plain",
        action="store_true",
        help="Output as plain key-value text (no colour)",
    )
    scn.add_argument(
        "--top-k",
        type=_positive_int,
        default=None,
        help="Max number of matches to return (default: 3)",
    )
    scn.add_argument(
        "--threshold",
        type=_unit_float,
        default=None,
        help="Min pipeline score for inclusion (default: 0.50)",
    )
    scn.add_argument(
        "--db-root",
        type=Path,
        default=None,
        help="Override the provenance database root directory",
    )
    scn.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override the default cache directory",
    )
    scn.add_argument(
        "--no-cache",
        dest="no_cache",
        action="store_true",
        help="Disable feature caching",
    )
    scn.add_argument(
        "--timing",
        dest="show_timing",
        action="store_true",
        help="Show phase-level timing breakdown (extraction, lookup)",
    )
    scn.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        help="Allow execution of model-hosted Python code (config/tokenizer). "
        "Use only with models you trust.",
    )

    dl = sub.add_parser(
        "download-deepsignals-fingerprint",
        help="Download deep-signal weight fingerprints from Hugging Face",
    )
    dl.add_argument(
        "--db-root",
        type=Path,
        default=None,
        help="Override the provenance database root directory",
    )
    dl.add_argument(
        "--update",
        action="store_true",
        help="Re-download and replace existing fingerprints with latest",
    )
    dl.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip SHA-256 integrity check after download",
    )
    dl.add_argument(
        "--status",
        action="store_true",
        help="Show current deep-signals installation status and exit",
    )

    return parser


def _run_with_spinner(label: str, func, *args, use_json: bool = False, **kwargs):  # type: ignore[no-untyped-def]
    """Run *func* with a Rich spinner on stderr.

    When *use_json* is ``True`` the spinner is suppressed so stdout
    stays machine-readable.
    """
    if use_json:
        return func(*args, **kwargs)

    console = Console(stderr=True)
    with console.status(
        f"[bold cyan]{label}[/]",
        spinner="dots",
    ) as status:
        return func(
            *args,
            **kwargs,
            on_phase=lambda phase: status.update(f"[bold cyan]{label}[/]  {phase}"),
        )


def _safe_run(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call *fn* and translate exceptions to stderr messages + exit code 1."""
    try:
        return fn(*args, **kwargs)
    except ProvenanceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return None


def _run_compare(args: argparse.Namespace) -> int:
    """Execute the ``compare`` subcommand."""
    from provenancekit.core.scanner import ModelProvenanceScanner

    kwargs: dict[str, Any] = {}
    if args.cache_dir is not None:
        kwargs["cache_dir"] = args.cache_dir
    if getattr(args, "trust_remote_code", False):
        kwargs["trust_remote_code"] = True
    settings = Settings(**kwargs)

    cache: CacheService | NullCache = (
        NullCache() if args.no_cache else CacheService(cache_dir=settings.cache_dir)
    )
    scanner = ModelProvenanceScanner(settings=settings, cache=cache)

    use_json = getattr(args, "output_json", False)
    result = _safe_run(
        _run_with_spinner,
        "Comparing models…",
        scanner.compare,
        args.model_a,
        args.model_b,
        use_json=use_json,
    )
    if result is None:
        return 1

    if args.output_json:
        output = format_json(result)
    elif args.output_plain:
        output = format_plain(result, include_timing=args.show_timing)
    else:
        output = format_table(result, include_timing=args.show_timing)

    print(output, end="")
    return 0


def _run_scan(args: argparse.Namespace) -> int:
    """Execute the ``scan`` subcommand."""
    from provenancekit.core.scanner import ModelProvenanceScanner

    settings_kwargs: dict[str, Any] = {}
    if args.cache_dir is not None:
        settings_kwargs["cache_dir"] = args.cache_dir
    if args.db_root is not None:
        settings_kwargs["db_root"] = args.db_root
    if getattr(args, "trust_remote_code", False):
        settings_kwargs["trust_remote_code"] = True
    settings = Settings(**settings_kwargs)

    cache: CacheService | NullCache = (
        NullCache() if args.no_cache else CacheService(cache_dir=settings.cache_dir)
    )
    scanner = ModelProvenanceScanner(settings=settings, cache=cache)

    use_json = getattr(args, "output_json", False)
    result = _safe_run(
        _run_with_spinner,
        "Scanning model…",
        scanner.scan,
        args.model_id,
        top_k=args.top_k,
        threshold=args.threshold,
        use_json=use_json,
    )
    if result is None:
        return 1

    if args.output_json:
        output = format_scan_json(result)
    elif args.output_plain:
        output = format_scan_plain(result, include_timing=args.show_timing)
    else:
        output = format_scan_table(result, include_timing=args.show_timing)

    print(output, end="")

    if not has_deep_signals(settings.db_root):
        print(
            "\nHint: Deep-signal fingerprints not installed. "
            "Run `provenancekit download-deepsignals-fingerprint` "
            "for weight-level matching.",
            file=sys.stderr,
        )

    return 0


def _run_download(args: argparse.Namespace) -> int:
    """Execute the ``download-deepsignals-fingerprint`` subcommand."""
    settings_kwargs: dict[str, Any] = {}
    if args.db_root is not None:
        settings_kwargs["db_root"] = args.db_root
    settings = Settings(**settings_kwargs)

    if args.status:
        return show_deep_signals_status(settings.db_root)

    if args.no_verify and not settings.dev_mode:
        print(
            "Error: --no-verify is disabled outside dev mode. "
            "Set PROVENANCEKIT_DEV_MODE=true to use it for local development only.",
            file=sys.stderr,
        )
        return 1

    if args.no_verify:
        print(
            "WARNING: --no-verify skips SHA-256 integrity validation. "
            "Use only in local development.",
            file=sys.stderr,
        )

    return download_deep_signals(
        settings.db_root,
        update=args.update,
        verify=not args.no_verify,
        settings=settings,
    )


def main() -> None:
    """CLI entry point registered via ``[project.scripts]``."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    configure_logging(getattr(args, "verbose", False))

    if args.command == "compare":
        sys.exit(_run_compare(args))
    elif args.command == "scan":
        sys.exit(_run_scan(args))
    elif args.command == "download-deepsignals-fingerprint":
        sys.exit(_run_download(args))
    else:
        parser.print_help()
        sys.exit(1)

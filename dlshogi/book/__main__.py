"""Command-line entry point for the PUCT Book Builder.

Invoked as ``python -m dlshogi.book <command>``. Implements the full
startup flow from design.md's "Error Handling" mermaid flowchart (nodes
C1 through C17), multi-process search spawning, and command dispatch to
Search_Coordinator, Value_Propagator, Book_Exporter, and the Terashock
import.

**No side effects at import time**: this module only defines functions,
and ``main()`` runs only under the ``__name__ == "__main__"`` guard at
the bottom of the file.

Requirements traced: 4.10, 4.13, 10.5, 10.6, 11.1, 13.1, 13.4, 13.5,
13.8, 13.9.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

_LOG = logging.getLogger("dlshogi.book")

# Shutdown budget (design.md: "within 60 s").
_SHUTDOWN_BUDGET_S = 60.0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with its subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m dlshogi.book",
        description="PUCT Book Builder: offline opening-book generation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- search --
    search_parser = subparsers.add_parser(
        "search", help="Run PUCT search to grow the book graph."
    )
    search_parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML configuration file.",
    )
    search_parser.add_argument(
        "--gpu-count", type=int, default=1,
        help="Number of GPU processes to spawn (default: 1).",
    )
    search_parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the ONNX model file.",
    )
    search_parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Enable verbose progress logging.",
    )
    search_parser.add_argument(
        "--log-path", type=str, default=None,
        help="Path for the log output file.",
    )

    # -- propagate --
    prop_parser = subparsers.add_parser(
        "propagate", help="Run the negamax value propagation pass."
    )
    prop_parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML configuration file.",
    )

    # -- export --
    export_parser = subparsers.add_parser(
        "export", help="Export the book graph to Apery and YaneuraOu formats."
    )
    export_parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML configuration file.",
    )
    export_parser.add_argument(
        "--apery", type=str, default=None,
        help="Output path for the Apery binary book file.",
    )
    export_parser.add_argument(
        "--yaneuraou", type=str, default=None,
        help="Output path for the YaneuraOu .db text file.",
    )

    # -- import-terashock --
    import_parser = subparsers.add_parser(
        "import-terashock", help="Import a YaneuraOu Terashock book."
    )
    import_parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML configuration file.",
    )

    return parser


# ---------------------------------------------------------------------------
# Configuration loading and validation (C1-C3)
# ---------------------------------------------------------------------------


def _load_and_validate_config(
    config_path: str, *, process_count: int = 1
) -> "BookConfig":
    """C1-C3: load, validate, and log the configuration.

    On validation failure, reports every absent name and every out-of-range
    name/value/range to stderr and exits with code 1 (Requirement 13.2,
    13.3, 13.5).
    """
    from dlshogi.book.config import (
        BookConfig,
        ConfigValidationError,
        load_yaml,
        redact_mapping,
    )

    # C1: Load configuration.
    try:
        raw = load_yaml(config_path)
    except (OSError, ValueError) as exc:
        _LOG.error("Failed to load configuration from %s: %s", config_path, exc)
        sys.exit(1)

    # C2: Validate all required values present and in range.
    try:
        cfg = BookConfig.load(raw, process_count=process_count)
    except ConfigValidationError as exc:
        _LOG.error("Configuration validation failed: %s", exc)
        sys.exit(1)

    # C3: Log configuration with credentials redacted.
    redacted = cfg.to_loggable_dict()
    _LOG.info("Configuration loaded: %s", redacted)

    return cfg


# ---------------------------------------------------------------------------
# C4: Import dlshogi.cppshogi
# ---------------------------------------------------------------------------


def _import_cppshogi() -> None:
    """C4: Import the C++ extension which runs initTable, initZobrist, etc.

    The extension's module-level ``init()`` call performs:
    ``initTable(); Position::initZobrist(); HuffmanCodedPos::init(); Book::init();``
    """
    try:
        import dlshogi.cppshogi  # noqa: F401
    except ImportError as exc:
        _LOG.error(
            "Failed to import dlshogi.cppshogi: %s. "
            "Ensure the C++ extension is built (see dlshogi/book/README.md).",
            exc,
        )
        sys.exit(1)
    _LOG.info("dlshogi.cppshogi imported successfully")


# ---------------------------------------------------------------------------
# C5-C9: Connect and ensure schema
# ---------------------------------------------------------------------------


async def _connect_and_ensure_schema(
    cfg: "BookConfig", role: str, *, read_only: bool = False
) -> "NodeStore":
    """C5-C9: Connect to PostgreSQL and run schema management.

    When ``read_only=True`` (child processes), schema creation/repair is
    skipped — only version/fingerprint verification is performed.
    """
    from dlshogi.book.node_store import (
        NodeStore,
        NodeStoreConnectionError,
        SchemaCreationError,
        SchemaVersionMismatchError,
    )

    store = NodeStore(cfg, role=role)

    # C5: Connect with the retry schedule.
    try:
        await store.connect()
    except NodeStoreConnectionError as exc:
        _LOG.error(
            "PostgreSQL connection exhausted retries: %s (Requirement 2.3)", exc
        )
        sys.exit(1)

    # C6-C9: Schema presence check, creation/repair, version/fingerprint.
    try:
        await store.ensure_schema()
    except SchemaCreationError as exc:
        _LOG.error("Schema creation failed: %s (Requirement 2.6)", exc)
        sys.exit(1)
    except SchemaVersionMismatchError as exc:
        _LOG.error(
            "Schema version or Zobrist fingerprint mismatch: %s (Requirement 2.8)",
            exc,
        )
        sys.exit(1)

    return store


# ---------------------------------------------------------------------------
# C11: Root position check
# ---------------------------------------------------------------------------


async def _check_root_mismatch(store: "NodeStore") -> None:
    """C11: If any node rows exist but no root row, report and exit (Req 10.8)."""
    has_nodes = await store.any_node_rows()
    if has_nodes:
        has_root = await store.root_node_exists()
        if not has_root:
            _LOG.error(
                "Root mismatch: book_node has rows but no Root_Position row "
                "exists. The configured Root_Position does not match the "
                "graph's root. (Requirement 10.8)"
            )
            sys.exit(1)


# ---------------------------------------------------------------------------
# C12-C15: Terashock import/verify
# ---------------------------------------------------------------------------


async def _handle_terashock(
    store: "NodeStore", cfg: "BookConfig", command: str
) -> Optional["TerashockIndex"]:
    """C12-C15: Check Terashock configuration and import if needed.

    Returns a TerashockIndex if configured, otherwise None.
    For `import-terashock` command, rejects if no path is configured.
    """
    from dlshogi.book.terashock import (
        TerashockIndex,
        check_terashock_source,
        import_terashock,
    )

    # C12: Terashock path configured?
    if cfg.terashock_book is None:
        # C15: No Terashock_Index; import-terashock rejected.
        if command == "import-terashock":
            _LOG.error(
                "No Terashock_Book path configured; import-terashock is "
                "rejected (Requirement 13.8, 13.9)"
            )
            sys.exit(1)
        return None

    # C13: Readable and yields >= 1 entry?
    # check_terashock_source returns True if import is needed.
    needs_import = await check_terashock_source(store.pool, cfg)

    if needs_import or command == "import-terashock":
        # C14: Import or verify terashock_entry.
        _LOG.info("Importing Terashock book from %s...", cfg.terashock_book)
        stats = await import_terashock(store.pool, cfg)
        _LOG.info(
            "Terashock import complete: %d entries, %d duplicates",
            stats.entry_count,
            stats.duplicate_count,
        )

    # Build the TerashockIndex for search use.
    if command == "search":
        terashock_idx = TerashockIndex(
            store.pool, cache_budget_bytes=cfg.cache_budget
        )
        return terashock_idx

    return None


# ---------------------------------------------------------------------------
# search: child process entry point
# ---------------------------------------------------------------------------


def _search_child_process(
    config_path: str,
    gpu_id: int,
    model_path: str,
    verbose: bool,
    log_path: Optional[str],
) -> None:
    """Entry point for a child search process (one per GPU).

    Repeats the read-only half of the startup checks (C4-C9), then runs
    the SearchCoordinator on its assigned GPU.
    """
    _setup_logging()
    _LOG.info("Search child process starting (gpu_id=%d, pid=%d)", gpu_id, os.getpid())

    # C1-C3: Load and validate config.
    from dlshogi.book.config import BookConfig, ConfigValidationError, load_yaml

    try:
        raw = load_yaml(config_path)
        cfg = BookConfig.load(raw)
    except (OSError, ValueError, ConfigValidationError) as exc:
        _LOG.error("Child process config validation failed: %s", exc)
        sys.exit(1)

    # C4: Import cppshogi.
    _import_cppshogi()

    # Run the async startup and search.
    asyncio.run(_search_child_async(cfg, gpu_id, model_path, verbose, log_path))


async def _search_child_async(
    cfg: "BookConfig",
    gpu_id: int,
    model_path: str,
    verbose: bool,
    log_path: Optional[str],
) -> None:
    """Async portion of the child search process."""
    from dlshogi.book.evaluator import Evaluator, OnnxRuntimeInferenceSession
    from dlshogi.book.keys import position_key_from_sfen
    from dlshogi.book.node_store import NodeStore
    from dlshogi.book.report import ProgressReporter
    from dlshogi.book.search import SearchCoordinator
    from dlshogi.book.terashock import TerashockIndex, check_terashock_source, import_terashock

    # C5-C9: Connect and verify schema (read-only: child does not create/repair).
    store = await _connect_and_ensure_schema(cfg, role="search", read_only=True)

    # C11: Root position check.
    await _check_root_mismatch(store)

    # C12-C15: Terashock (lookup only, no import from child).
    terashock_idx: Optional[TerashockIndex] = None
    if cfg.terashock_book is not None:
        terashock_idx = TerashockIndex(store.pool, cache_budget_bytes=cfg.cache_budget)

    # C16: Clear in-flight claims (each process clears independently).
    cleared = await store.clear_in_flight_claims_at_startup()
    _LOG.info(
        "Cleared %d stale in_flight_claim rows (gpu_id=%d, Requirement 10.2)",
        cleared, gpu_id,
    )

    # Create the Evaluator with a real ONNX session.
    session = OnnxRuntimeInferenceSession(
        model_path,
        device_id=gpu_id,
        fixed_batch_size=cfg.batch_size,
    )
    evaluator = Evaluator(
        session,
        batch_size=cfg.batch_size,
        batch_timeout_ms=cfg.batch_timeout,
    )
    evaluator.start()

    # Create the ProgressReporter.
    reporter = ProgressReporter(
        report_interval=cfg.report_interval,
        throughput_floor=cfg.throughput_floor,
        throughput_grace_period=cfg.throughput_grace_period,
        verbose=verbose,
        log_path=log_path,
        process_id=os.getpid(),
    )
    reporter.start()

    # Create and run the SearchCoordinator.
    coordinator = SearchCoordinator(
        config=cfg,
        node_store=store,
        evaluator=evaluator,
        terashock_index=terashock_idx,
    )

    try:
        await coordinator.run()
    finally:
        reporter.stop()
        await evaluator.stop()


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _run_search(args: argparse.Namespace) -> int:
    """Run the PUCT search command with multi-process GPU support."""
    _setup_logging()
    gpu_count = args.gpu_count
    config_path = args.config
    model_path = args.model
    verbose = args.verbose
    log_path = args.log_path

    # C1-C3: Load and validate config (parent does the full startup).
    cfg = _load_and_validate_config(config_path, process_count=gpu_count)

    # C4: Import cppshogi.
    _import_cppshogi()

    if gpu_count <= 1:
        # Single-process: run directly in this process.
        return asyncio.run(
            _search_single_process(cfg, model_path, verbose, log_path)
        )
    else:
        # Multi-process: parent performs schema setup, then spawns children.
        return asyncio.run(
            _search_parent_process(cfg, config_path, gpu_count, model_path, verbose, log_path)
        )


async def _search_single_process(
    cfg: "BookConfig",
    model_path: str,
    verbose: bool,
    log_path: Optional[str],
) -> int:
    """Single-process search (gpu_count=1): full startup + search loop."""
    from dlshogi.book.evaluator import Evaluator, OnnxRuntimeInferenceSession
    from dlshogi.book.report import ProgressReporter
    from dlshogi.book.search import SearchCoordinator

    # C5-C9: Connect and ensure schema.
    store = await _connect_and_ensure_schema(cfg, role="search")

    # C11: Root position check.
    await _check_root_mismatch(store)

    # C12-C15: Terashock.
    terashock_idx = await _handle_terashock(store, cfg, "search")

    # C16: Clear in-flight claims.
    cleared = await store.clear_in_flight_claims_at_startup()
    _LOG.info("Cleared %d stale in_flight_claim rows (Requirement 10.2)", cleared)

    # Create the Evaluator.
    session = OnnxRuntimeInferenceSession(
        model_path,
        device_id=0,
        fixed_batch_size=cfg.batch_size,
    )
    evaluator = Evaluator(
        session,
        batch_size=cfg.batch_size,
        batch_timeout_ms=cfg.batch_timeout,
    )
    evaluator.start()

    # Create the ProgressReporter.
    reporter = ProgressReporter(
        report_interval=cfg.report_interval,
        throughput_floor=cfg.throughput_floor,
        throughput_grace_period=cfg.throughput_grace_period,
        verbose=verbose,
        log_path=log_path,
        process_id=os.getpid(),
    )
    reporter.start()

    # Create and run the SearchCoordinator.
    coordinator = SearchCoordinator(
        config=cfg,
        node_store=store,
        evaluator=evaluator,
        terashock_index=terashock_idx,
    )

    try:
        await coordinator.run()
    finally:
        reporter.stop()
        await evaluator.stop()

    return 0


async def _search_parent_process(
    cfg: "BookConfig",
    config_path: str,
    gpu_count: int,
    model_path: str,
    verbose: bool,
    log_path: Optional[str],
) -> int:
    """Multi-process search: parent does schema setup, spawns children.

    C17: Spawn one process per GPU. The parent forwards stop signals to
    every child inside one 60 s budget.
    """
    # C5-C9: Parent performs full startup including schema creation/repair.
    store = await _connect_and_ensure_schema(cfg, role="search")

    # C11: Root position check.
    await _check_root_mismatch(store)

    # C12-C15: Terashock import (only parent imports).
    await _handle_terashock(store, cfg, "search")

    # C16: Clear in-flight claims (parent clears before spawning).
    cleared = await store.clear_in_flight_claims_at_startup()
    _LOG.info("Cleared %d stale in_flight_claim rows (Requirement 10.2)", cleared)

    _LOG.info("Spawning %d search child processes...", gpu_count)

    # Spawn one child process per GPU.
    children: list[multiprocessing.Process] = []
    for gpu_id in range(gpu_count):
        proc = multiprocessing.Process(
            target=_search_child_process,
            args=(config_path, gpu_id, model_path, verbose, log_path),
            name=f"book-search-gpu{gpu_id}",
            daemon=False,
        )
        proc.start()
        children.append(proc)
        _LOG.info("Started child process pid=%d for gpu_id=%d", proc.pid, gpu_id)

    # Install signal handlers to forward stop signals to children.
    stop_requested = False

    def _forward_signal(signum: int, frame: object) -> None:
        nonlocal stop_requested
        if stop_requested:
            return
        stop_requested = True
        sig_name = signal.Signals(signum).name
        _LOG.info(
            "Received %s, forwarding to %d child processes...",
            sig_name, len(children),
        )
        for child in children:
            if child.is_alive() and child.pid is not None:
                try:
                    os.kill(child.pid, signal.SIGTERM)
                except OSError:
                    pass

    signal.signal(signal.SIGINT, _forward_signal)
    signal.signal(signal.SIGTERM, _forward_signal)

    # Wait for all children within the 60 s budget.
    deadline = time.monotonic() + _SHUTDOWN_BUDGET_S
    all_ok = True
    for child in children:
        remaining = max(0.1, deadline - time.monotonic())
        child.join(timeout=remaining)
        if child.is_alive():
            _LOG.warning(
                "Child pid=%d did not exit within the shutdown budget; "
                "terminating forcefully.",
                child.pid,
            )
            child.terminate()
            child.join(timeout=5.0)
            all_ok = False
        elif child.exitcode != 0:
            _LOG.error(
                "Child pid=%d exited with code %d", child.pid, child.exitcode
            )
            all_ok = False

    if not all_ok:
        _LOG.error("One or more child processes exited abnormally.")
        return 1

    _LOG.info("All child processes exited successfully.")
    return 0


def _run_propagate(args: argparse.Namespace) -> int:
    """Run the negamax value propagation pass."""
    _setup_logging()
    config_path = args.config

    # C1-C3: Load and validate config.
    cfg = _load_and_validate_config(config_path)

    # C4: Import cppshogi.
    _import_cppshogi()

    return asyncio.run(_propagate_async(cfg))


async def _propagate_async(cfg: "BookConfig") -> int:
    """Async portion of the propagate command."""
    from dlshogi.book.keys import position_key_from_sfen
    from dlshogi.book.propagate import propagate

    # C5-C9: Connect and ensure schema.
    store = await _connect_and_ensure_schema(cfg, role="propagate")

    # C11: Root position check.
    await _check_root_mismatch(store)

    # Run propagation.
    root_key = position_key_from_sfen(cfg.root_position)
    stats = await propagate(store, root_key, cfg)

    _LOG.info(
        "Propagation complete: nodes=%d, root_value=%.6f, "
        "draw_revisits=%d, unevaluated_leaves=%d, "
        "path_cutoffs=%d, below_threshold_edges=%d",
        stats.nodes,
        stats.root_value,
        stats.draw_revisits,
        stats.unevaluated_leaves,
        stats.path_cutoffs,
        stats.below_threshold_edges,
    )
    return 0


def _run_export(args: argparse.Namespace) -> int:
    """Export the book graph to Apery and/or YaneuraOu format."""
    _setup_logging()
    config_path = args.config
    apery_out = Path(args.apery) if args.apery else None
    yaneuraou_out = Path(args.yaneuraou) if args.yaneuraou else None

    if apery_out is None and yaneuraou_out is None:
        _LOG.error("At least one of --apery or --yaneuraou must be specified.")
        return 1

    # C1-C3: Load and validate config.
    cfg = _load_and_validate_config(config_path)

    # C4: Import cppshogi.
    _import_cppshogi()

    return asyncio.run(_export_async(cfg, apery_out, yaneuraou_out))


async def _export_async(
    cfg: "BookConfig",
    apery_out: Optional[Path],
    yaneuraou_out: Optional[Path],
) -> int:
    """Async portion of the export command."""
    from dlshogi.book.export import export_book

    # C5-C9: Connect and ensure schema.
    store = await _connect_and_ensure_schema(cfg, role="export")

    # C11: Root position check.
    await _check_root_mismatch(store)

    # Run export.
    try:
        counts = await export_book(store, cfg, apery_out, yaneuraou_out)
    except OSError as exc:
        _LOG.error("Export failed (Requirement 12.10): %s", exc)
        return 1

    _LOG.info(
        "Export complete: apery_records=%d, yaneuraou_entries=%d, "
        "excluded_edges=%d, apery_key_multi_node=%d",
        counts.apery_records,
        counts.yaneuraou_entries,
        counts.excluded_edges,
        counts.apery_key_multi_node,
    )
    return 0


def _run_import_terashock(args: argparse.Namespace) -> int:
    """Import a YaneuraOu Terashock book."""
    _setup_logging()
    config_path = args.config

    # C1-C3: Load and validate config.
    cfg = _load_and_validate_config(config_path)

    # Reject if no Terashock path configured (Requirement 13.8, 13.9).
    if cfg.terashock_book is None:
        _LOG.error(
            "No Terashock_Book path configured; import-terashock is "
            "rejected (Requirement 13.8, 13.9)"
        )
        return 1

    # C4: Import cppshogi.
    _import_cppshogi()

    return asyncio.run(_import_terashock_async(cfg))


async def _import_terashock_async(cfg: "BookConfig") -> int:
    """Async portion of the import-terashock command."""
    from dlshogi.book.terashock import import_terashock

    # C5-C9: Connect and ensure schema.
    store = await _connect_and_ensure_schema(cfg, role="search")

    # C11: Root position check.
    await _check_root_mismatch(store)

    # Run import.
    _LOG.info("Importing Terashock book from %s...", cfg.terashock_book)
    stats = await import_terashock(store.pool, cfg)
    _LOG.info(
        "Terashock import complete: entries_imported=%d, duplicates=%d",
        stats.entry_count,
        stats.duplicate_count,
    )
    return 0


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    """Configure the root logger for the dlshogi.book package."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "search": _run_search,
        "propagate": _run_propagate,
        "export": _run_export,
        "import-terashock": _run_import_terashock,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

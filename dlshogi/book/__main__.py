"""Command-line entry point for the PUCT Book Builder.

Invoked as ``python -m dlshogi.book <command>``. Defines the ``argparse``
subcommand skeleton for ``search``, ``propagate``, ``export``, and
``import-terashock``. No side effects occur at import time: this module only
defines functions, and ``main()`` runs only under the ``__main__`` guard at
the bottom of the file.

The subcommand bodies are implemented by later tasks (Search_Coordinator by
task 13, Value_Propagator by task 14, Book_Exporter by task 15, and the
Terashock import by task 12.3).
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with its subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m dlshogi.book",
        description="PUCT Book Builder: offline opening-book generation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("search", help="Run PUCT search to grow the book graph.")
    subparsers.add_parser("propagate", help="Run the negamax value propagation pass.")
    subparsers.add_parser(
        "export", help="Export the book graph to Apery and YaneuraOu formats."
    )
    subparsers.add_parser(
        "import-terashock", help="Import a YaneuraOu Terashock book."
    )

    return parser


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


def _run_search(args: argparse.Namespace) -> int:
    raise NotImplementedError("search command is implemented by task 13")


def _run_propagate(args: argparse.Namespace) -> int:
    raise NotImplementedError("propagate command is implemented by task 14")


def _run_export(args: argparse.Namespace) -> int:
    raise NotImplementedError("export command is implemented by task 15")


def _run_import_terashock(args: argparse.Namespace) -> int:
    raise NotImplementedError(
        "import-terashock command is implemented by task 12.3"
    )


if __name__ == "__main__":
    raise SystemExit(main())

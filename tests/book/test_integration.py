"""Binding build-and-import gate test (task 2.3).

This is the gate that declares the task 2 prerequisite -- the 128-bit
Position_Key Cython binding -- complete: it asserts that the extended
``dlshogi.cppshogi`` module actually builds and imports in this
environment, that the four functions task 2.1/2.2 add are present, and
that ``zobrist_fingerprint()`` reproduces the golden value recorded in
``tests/book/fixtures/zobrist_fingerprint.txt``.

This test is deliberately thin. The substance of Properties 8 and 10
(Position_Key depends only on the Board_State; incremental child key
equals the recomputed key) is exercised in depth by
``tests/book/test_keys.py`` (tasks 3.2/3.3), including golden Position_Key
vectors from ``tests/book/fixtures/position_keys.json`` and a
cross-process stability check. That file already implicitly proves the
binding builds and imports -- it cannot run otherwise -- so this test's
job is narrower: to be the one file whose name and docstring make that
gate explicit and independently checkable, and to be the file task 19.1
extends with the scale integration test once the rest of the design is
built.
"""

from __future__ import annotations

from pathlib import Path

from dlshogi import cppshogi
from dlshogi.book.keys import zobrist_fingerprint

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_ZOBRIST_FINGERPRINT_FIXTURE = _FIXTURES_DIR / "zobrist_fingerprint.txt"


def test_the_four_binding_functions_are_importable():
    """The four free functions added in task 2.1/2.2 are present on the
    built and imported ``dlshogi.cppshogi`` extension module."""
    for name in (
        "position_key_from_sfen",
        "position_keys_after",
        "zobrist_fingerprint",
        "apery_book_key_from_sfen",
    ):
        assert hasattr(cppshogi, name), f"dlshogi.cppshogi.{name} is missing"
        assert callable(getattr(cppshogi, name))


def test_zobrist_fingerprint_matches_the_golden_value():
    """``zobrist_fingerprint()`` against the built extension reproduces the
    golden value pinned in ``tests/book/fixtures/zobrist_fingerprint.txt``.

    A mismatch means Zobrist table initialization changed in a way that
    would silently re-key every Position_Key already stored in a book
    database (Requirement 3.4), so this is checked directly here rather
    than only through ``tests/book/test_keys.py``'s consumption of the
    same fixture.
    """
    golden_fingerprint = int(_ZOBRIST_FINGERPRINT_FIXTURE.read_text().strip())
    assert zobrist_fingerprint() == golden_fingerprint

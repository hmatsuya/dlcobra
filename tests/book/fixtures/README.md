# tests/book/fixtures/

Static test data consumed by the `tests/book/` property and example test
suite. Nothing in this directory is generated at test-collection time;
each corpus is either a small hand-curated set of real Shogi positions that
random generation would not reliably reproduce, or a golden-value fixture
that pins a specific implementation output so a future regression is
caught even if the generator that originally exercised it changes.

Files land here as the tasks that need them are implemented. This README
is kept up to date with what each corpus is for and which task/property
populates and consumes it.

## Planned corpora

- **`zobrist_fingerprint.txt`** (task 2.3) -- the golden `zobrist_fingerprint()`
  value recorded once against the extended `dlshogi.cppshogi` binding.
  `tests/book/test_integration.py`'s binding build-and-import gate test
  compares the running binding's fingerprint against this file, so an
  accidental change to Zobrist table initialization order is caught
  immediately rather than silently re-keying every Position_Key already
  stored in a book database (see Requirement 3.4 and design.md's
  *Position_Key from Python* section).

- **`position_keys.json`** (task 2.3) -- golden `(SFEN, key_hi, key_lo)`
  triples covering the initial position, representative mid-game
  positions, drop and promotion positions, and pairs of positions that
  differ only in the non-moving side's hand (the case that Apery's 64-bit
  `Book::bookKey` gets wrong and that Requirement 3.2 requires the 128-bit
  Position_Key to get right). Used by Properties 8 and 10 as `@example`
  pins and by the binding gate test.

- **Perpetual-check positions** (task 11.4) -- real SFEN positions for
  four-fold repetition under continuous check by the mover only, by the
  opponent only, by both sides, and by neither side. These back Property
  23's repetition-classification truth table (Requirements 8.3, 8.4) with
  positions that `hypothesis`'s random position generator does not
  reliably produce on its own, because perpetual check sequences are a
  narrow slice of the legal-position space.

- **`.db` text corpora** (tasks 5.3, 5.4) -- small hand-written YaneuraOu
  `.db` files, including deliberately malformed lines (blank lines,
  comments, an `opponent reply` field of `none`, and lines that violate
  the five-field candidate-move shape), used to check Book_DB_Parser's
  line-classification and noise-tolerance behavior (Requirement 6) against
  fixed, human-reviewable input rather than only against generated input.

- **Entering-king (nyugyoku) boundary positions** (task 11.3, alongside the
  `@composite` generator in `tests/book/strategies.py`) -- positions that
  straddle the point-count and piece-count boundaries of the declaration-win
  rule, used to check the independently written declaration-win predicate
  against `cshogi`'s `board.is_nyugyoku()` (Requirement 8.6).

Each corpus above is added by the task that first consumes it; this file's
"Planned corpora" list is updated to "present" (with a one-line usage note
per file already added) as those tasks land, so this README never falls
out of sync with the directory's actual contents.

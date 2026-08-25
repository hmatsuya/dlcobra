# Implementation Plan: PUCT Book Builder

## Overview

The implementation follows the design's dependency order: the 128-bit Position_Key Cython binding
lands first, because nothing can be built or tested against real positions without it; then the pure
modules that need no database and no GPU (`keys.py`, `packed_edge.py`, `config.py`, `book_db.py`);
then the PostgreSQL schema and `node_store.py`; then the Evaluator against a stub session, the
Repetition_Resolver, the Prior_Mixer and Terashock_Index; then `search.py`, `propagate.py`,
`export.py`, and `report.py`; and finally the CLI wiring and the integration work that measures the
storage design at scale.

The language is Python (the design specifies it throughout), with C++ additions confined to
`cppshogi/python_module.{h,cpp}` and C confined to `dlshogi/book/pgext/puct_edge.c`.

All 48 correctness properties of the design are implemented, each by exactly one property-based test,
with no collapsing and no renumbering. Property test files follow the layout in the design's Testing
Strategy. Properties 1-5, 11, 13-16, 19, 28-40, and 47 require a real scratch PostgreSQL database and
carry the `db` marker so `pytest -m "not db"` stays fast. Properties 3, 15, 18, 43, and 44 are
`hypothesis.stateful.RuleBasedStateMachine` tests. Property 9 runs as a single 1,000,000-position
example, not 100 short ones.

## Tasks

- [x] 1. Project scaffolding, dependencies, and test harness
  - [x] 1.1 Register the package and declare dependencies
    - Add `'dlshogi.book'` to `packages` in `setup.py` alongside `'dlshogi.network'` and `'dlshogi.utils'`
    - Add to `Pipfile` `[packages]`: `asyncpg = "==0.30.0"`, `onnxruntime-gpu = "==1.20.1"`, `numpy = "*"`
    - Add to `Pipfile` `[dev-packages]`: `pytest = "==8.3.4"`, `pytest-asyncio = "==0.25.0"`, `hypothesis = "==6.122.3"`
    - Record the required system packages (`libpq-dev`, `postgresql-server-dev-<N>` for the PGXS build, PostgreSQL server 14 or later) in a short `dlshogi/book/README.md` prerequisites note
    - _Requirements: 13.1_

  - [x] 1.2 Create the `dlshogi/book/` package skeleton
    - `dlshogi/book/__init__.py` exporting the public names `BookConfig`, `NodeStore`, `PACKED_EDGE`
    - `dlshogi/book/__main__.py` with the `argparse` subcommand skeleton `search` / `propagate` / `export` / `import-terashock`, no side effects at import time
    - Create the empty module files the later tasks fill so imports resolve: `keys.py`, `packed_edge.py`, `config.py`, `node_store.py`, `search.py`, `repetition.py`, `evaluator.py`, `prior_mixer.py`, `book_db.py`, `terashock.py`, `propagate.py`, `export.py`, `report.py`, `sql/`, `pgext/`
    - _Requirements: 13.4_

  - [x] 1.3 Create the test harness configuration and fixtures
    - `pytest.ini` with `asyncio_mode = auto`, a registered `db` marker, and `testpaths = tests`
    - `tests/book/conftest.py`: session-scoped scratch-database fixture creating `puct_test_<pid>` and dropping it at teardown, a per-example truncate-and-repopulate fixture, and a virtual-clock event loop fixture whose `time()` the test advances so no test sleeps on the wall clock
    - `tests/book/fixtures/` directory with a README stating what each fixture corpus is for
    - _Requirements: 13.1_

- [ ] 2. 128-bit Position_Key Cython binding (hard prerequisite)
  - [x] 2.1 Add the four free functions to the C++ python module
    - In `cppshogi/python_module.h` declare `__position_key_from_sfen`, `__position_keys_after`, `__zobrist_fingerprint`, `__apery_book_key_from_sfen` in the style of the existing free functions
    - Implement them in `cppshogi/python_module.cpp` over `Position::getBoardKey()`, `Position::getHandKey()`, and `Position::getKeyAndBoardKeyAfter(Move)`; `__position_keys_after` sets a `Position` from the SFEN once then iterates the supplied `move16` array
    - Extend the existing `init()` with `Book::init()`, keeping the order `initTable(); Position::initZobrist(); HuffmanCodedPos::init(); Book::init();` so nothing draws from `g_mt64bit` before `initZobrist()`
    - No new build target: `setup.py` already compiles `cppshogi/position.cpp`, `book.cpp`, and `search.cpp` into `dlshogi.cppshogi`
    - _Requirements: 3.1, 3.2, 3.4, 3.7_

  - [x] 2.2 Add the Cython declarations and wrappers
    - In `dlshogi/cppshogi.pyx` add the matching `cdef extern from "python_module.h" nogil:` block and the Python wrappers `position_key_from_sfen`, `position_keys_after`, `zobrist_fingerprint`, `apery_book_key_from_sfen`
    - Wrappers take numpy buffers of the 16-byte key dtype and encode SFEN strings with `locale.getpreferredencoding()`, matching the existing pattern
    - _Requirements: 3.1, 3.7_

  - [ ]* 2.3 Write the binding build-and-import gate test
    - `tests/book/test_integration.py`: build and import the extended `dlshogi.cppshogi`, assert the four new functions are importable, assert `zobrist_fingerprint()` equals the golden value recorded in `tests/book/fixtures/zobrist_fingerprint.txt`
    - Create the golden Position_Key vector fixture `tests/book/fixtures/position_keys.json` as (SFEN, key_hi, key_lo) triples covering the initial position, mid-game positions, drop and promotion positions, and positions differing only in the non-moving side's hand
    - This test is the gate that declares the prerequisite complete
    - _Requirements: 3.1, 3.4_
    - Note: the golden fixtures (`tests/book/fixtures/zobrist_fingerprint.txt`, `tests/book/fixtures/position_keys.json`) were created and are consumed by `tests/book/test_keys.py` (task 3.2). The dedicated `tests/book/test_integration.py` build-and-import gate test itself remains unwritten.

- [x] 3. Position_Key module, packed edge codec, and shared strategies
  - [x] 3.1 Implement `dlshogi/book/keys.py`
    - `POSITION_KEY = np.dtype([("hi", "<u8"), ("lo", "<u8")])` with `assert POSITION_KEY.itemsize == 16`
    - `PositionKey` value type, `position_key(board)`, `position_key_from_sfen(sfen)`, `position_keys_after(sfen, moves16)` batched form, `apery_book_key(sfen)`, `zobrist_fingerprint()`
    - `side_to_move_is_white(key)` as the `key.hi & 1` test, with the reasoning comment that every Zobrist table entry has bit 0 cleared and `zobTurn_ == 1`
    - Two's-complement fold helpers for the signed `bigint` columns in both directions
    - _Requirements: 3.1, 3.2, 3.4, 3.7_

  - [x] 3.5 Implement `dlshogi/book/packed_edge.py`
    - `PACKED_EDGE` numpy dtype, 20 bytes, fields `move16 <u2`, `prior_q16 <u2`, `ts_depth u1`, `flags u1`, `ts_eval <i2`, `visit_count <u4`, `value_sum <f8`
    - `PACKED_TS_MOVE` numpy dtype, 11 bytes, fields `move16 <u2`, `reply16 <u2`, `eval <i2`, `depth u1`, `count <u4`
    - `encode_edges` writing ascending USI order with absent Terashock fields as canonical zeros, `decode_edges` as a zero-copy `np.frombuffer`, and `patch_edges` performing the client-side `searchsorted` visit/value patch used by the fallback backup path
    - Module-level assertions on `itemsize` and on the exact field offsets `[0, 2, 4, 5, 6, 8, 12]` and `[0, 2, 4, 6, 7]`
    - _Requirements: 1.2, 1.3, 1.7, 4.4, 7.2_

  - [x] 3.6 Write the packed edge layout and codec tests
    - `tests/book/test_packed_edge.py`: dtype itemsize and field-offset assertions for both dtypes, `isalignedstruct is False`, little-endian format assertions, and an encode/decode round trip including the ascending-USI order assertion and the canonical-zero encoding of absent Terashock fields
    - _Requirements: 1.2, 1.3, 1.7_
    - Note: writing this test uncovered a real bug in `encode_edges`/`patch_edges`: they sorted/searched by numeric `move16` rather than by USI text, which diverge whenever a promotion is present (promotion sets bit 14 of `move16`, which is not equivalent to lexicographic USI order). Fixed both functions in `dlshogi/book/packed_edge.py` to sort/index by `cshogi.move_to_usi(move16)`.

  - [x] 3.7 Write the shared hypothesis strategies
    - `tests/book/strategies.py` with the `@composite` strategies the design's property statements reuse: the self-play position strategy, the Position_Key strategy, the node-field strategy of Property 1, the `PACKED_EDGE` edge-array strategy of Property 12, the random-DAG graph strategy of Property 26 (transposition and back-edge rates, terminal and `eval_win_rate` masks, threshold-straddling visit counts), and the Terashock entry-list strategy of Property 20
    - _Requirements: 3.2, 4.2, 6.2, 9.1_
    - Note: only the self-play position strategy (`self_play_sfen`/`self_play_board_with_moves`, needed by tasks 3.2-3.4 and 5.3-5.4) is implemented so far. The node-field, PACKED_EDGE edge-array, random-DAG, and Terashock entry-list strategies are added by the later tasks (7.7, 13.4, 14.3, 5.3) that first need them.

  - [x] 3.2 Write property test for Position_Key stability
    - **Property 8: Position_Key depends only on the Board_State**
    - **Validates: Requirements 3.2, 3.4**
    - Include the subprocess recomputation so "across separate runs" is tested rather than assumed, and check the golden vector fixture and `zobrist_fingerprint()`
    - Implemented in `tests/book/test_keys.py`, backed by `tests/book/fixtures/position_keys.json` and `zobrist_fingerprint.txt`.

  - [x] 3.3 Write property test for the incremental child key
    - **Property 10: Incremental child key equals the recomputed key**
    - **Validates: Requirements 3.7**
    - Check every legal move of each drawn position, and assert the batched `position_keys_after` agrees element-wise with the single-move form
    - Implemented in `tests/book/test_keys.py`.

  - [x] 3.4 Write property test for key collision freedom
    - **Property 9: Position_Key is collision-free over the verification sample**
    - **Validates: Requirements 3.3**
    - One single example of 1,000,000 or more distinct Board_States under `@settings(max_examples=1, deadline=None)`, not 100 short examples
    - Implemented in `tests/book/test_keys.py` (no `@given`/`@settings`, since the sample is one fixed-seed deterministic example, not a hypothesis-generated one); combines self-play random walks with random hand-redistribution perturbations. Runs in ~12s.

- [x] 4. Configuration loading and validation
  - [x] 4.1 Implement `dlshogi/book/config.py`
    - `BookConfig` holding the nineteen values of Requirement 13 criterion 1
    - The Requirement 13 criterion 7 permitted-range table expressed **once** as a table of `(name, kind, low, high)` rows, consumed by both the validator and Property 41's generator; `Propagation_Visit_Threshold` (absolute count, 0 to 1,000,000) and `Export_Visit_Threshold` (ratio, 0 to 1) are distinct rows with distinct kinds
    - Collect every absent and every out-of-range value and report them together with name, supplied value, and permitted range, then exit before any schema work or record write
    - Credential redaction for the startup configuration log
    - The two sizing **warnings**, which are warnings and not range violations: `Throughput_Floor > 1500 * process_count` and `Batch_Size > 2 * Throughput_Floor * Batch_Timeout`, plus `Worker_Count < 2 * Batch_Size`
    - _Requirements: 13.1, 13.2, 13.3, 13.5, 13.6, 13.7_

  - [x] 4.2 Write property test for configuration validation
    - **Property 41: Configuration validation is exact**
    - **Validates: Requirements 13.2, 13.3, 13.5, 13.7**
    - The generator is driven from the range table in `config.py`, not from a hand-written list, with a 19-wide subset mask
    - Implemented in `tests/book/test_config_properties.py`, driven from `config.RANGE_TABLE`'s full row list (23 rows, per that module's documented deviation from the literal "19" count) rather than a hand-written list.

  - [x] 4.3 Write property test for credential redaction
    - **Property 42: Credentials are redacted**
    - **Validates: Requirements 13.6**
    - Implemented in `tests/book/test_config_properties.py`, including the substring-overlap case.

- [x] 5. YaneuraOu `.db` parsing and printing
  - [x] 5.1 Write the independent `.db` reference implementation
    - `tests/book/reference/book_db.py`: a parser and printer written **from the Requirement 6 text alone**, importing nothing from `dlshogi.book`, reviewed against the requirements rather than against the implementation
    - Both sides are Python now, so independence is deliberate rather than a consequence of a language boundary; state that in the module docstring
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_
    - Note: this reference keeps moves as plain text (no board-based legality resolution), a deliberate, documented scope difference from `dlshogi.book.book_db`'s move16-canonicalisation; see the module docstring.

  - [x] 5.2 Implement `dlshogi/book/book_db.py`
    - `TerashockMove`, `TerashockEntry`, `TerashockBook` dataclasses with moves stored as `move16` so the printer output is canonical
    - Book_DB_Parser as a line-oriented state machine with the classification order blank, `#YANEURAOU-DB2016 <version>`, other `#`, `sfen `, five-field candidate move, anything else; states *before any sfen line* and *inside an entry*
    - Book_DB_Printer emitting the header line, the `# NOE:<count>` line, `sfen ` lines, and five-field move lines with single-space separators and the literal `none` for an absent reply
    - Every rejection reports line number and content to the Progress_Reporter and keeps everything parsed so far
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.10, 6.11, 6.12_

  - [x] 5.3 Write property test for the entry-list round trip
    - **Property 20: YaneuraOu `.db` printer/parser round trip**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.6, 6.7, 6.8**
    - Cross-check against `tests/book/reference/book_db.py`, not only against the implementation's own inverse
    - Implemented in `tests/book/test_book_db.py`.

  - [x] 5.4 Write property test for the text round trip and noise tolerance
    - **Property 21: `.db` text round trip and noise tolerance**
    - **Validates: Requirements 6.4, 6.5, 6.9, 6.10, 6.11, 6.12**
    - Cross-check against `tests/book/reference/book_db.py`
    - Implemented in `tests/book/test_book_db.py`.

- [x] 6. Checkpoint - pure modules green without a database or a GPU
  - Ensure all tests pass, ask the user if questions arise.
  - `pytest -m "not db"` covers Properties 8, 9, 10, 20, 21, 41, 42 and the packed-edge layout assertions at this point
  - Result: `pytest -m "not db"` -> 76 passed, 0 failed, in ~24s. Fixed a real bug found while writing the packed-edge codec test (see task 3.6's note). The task 2.3 gate test itself (`tests/book/test_integration.py`) and the node-field/edge-array/graph/entry-list hypothesis strategies remain for their owning later tasks, per task 3.7's note.

- [x] 7. Schema management and Node_Store read path
  - [x] 7.1 Write `dlshogi/book/sql/schema.sql`
    - `book_meta`, `book_node`, `terashock_entry`, `in_flight_claim` DDL exactly as the design's Schema section states, including the `book_node_edges_len` check, `WITH (fillfactor = 70)`, and `ALTER COLUMN edges/sfen SET STORAGE MAIN`
    - Exactly one index on `book_node`, the primary key on `(key_hi, key_lo)`; no index on `apery_key`, `prop_epoch`, or any child key
    - A comment recording that `book_node` is deliberately **not** `UNLOGGED`, because `UNLOGGED` relations are truncated by crash recovery and that would contradict Requirements 2.3, 10.1, and 10.3
    - _Requirements: 1.1, 1.2, 2.4, 2.7_

  - [x] 7.2 Implement Node_Store connection and schema management
    - `dlshogi/book/node_store.py`: asyncpg pool with `max_size = min(Worker_Count, 64)`, connections acquired per statement rather than per descent
    - Schema creation within 300 s, schema version and Zobrist fingerprint recording and comparison, Root_Position recording and read-back, absent-element-only repair, and the connection retry schedule (5 s per attempt, 1 s doubling to a 60 s cap, `Connection_Retry_Limit` retries)
    - The lost-connection suspension seam: one catch point for `ConnectionDoesNotExistError` / `ConnectionResetError`, a suspension `asyncio.Event` every descent task waits on, resumption on reconnect
    - Set `synchronous_commit = off` on the search session and `on` for propagation and export sessions, and log the effective setting at startup
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 4.10_

  - [x] 7.6 Implement the Node_Store read path
    - `BookNodeView`, `GetResult` with `FOUND` / `ABSENT` / `FAILED` distinct, `get`, `get_many` over `(key_hi, key_lo) = ANY(...)`, and `get_many_terminal_eval` as the narrow `(terminal, eval_win_rate)` projection for below-threshold propagation children
    - `get` consults the node LRU, then the pending backup accumulator, then PostgreSQL; the absent path issues no INSERT
    - Node LRU with byte accounting bounded by Cache_Budget, evicting rather than failing
    - Edges returned as a zero-copy `np.frombuffer` view in ascending USI order
    - _Requirements: 1.3, 1.4, 1.5, 1.8_

  - [ ]* 7.3 Write property test for the connection retry schedule
    - **Property 6: Connection retry schedule**
    - **Validates: Requirements 2.2**

  - [ ]* 7.4 Write property test for schema version mismatch
    - **Property 7: Schema version mismatch is inert**
    - **Validates: Requirements 2.8**

  - [ ]* 7.5 Write property test for schema repair
    - **Property 5: Schema repair creates only what is absent**
    - **Validates: Requirements 2.5**
    - `db` marker; compare against a `pg_catalog` snapshot taken before the drop

  - [ ]* 7.7 Write property test for the Node_Store round trip
    - **Property 1: Node_Store round trip**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.7**
    - `db` marker; `@example` pins 0 edges, 600 edges, and `visit_count = 2**32-1`

  - [ ]* 7.8 Write property test for the absent result
    - **Property 2: Absent is absent**
    - **Validates: Requirements 1.4**
    - `db` marker

  - [ ]* 7.9 Write property test for the cache budget bound
    - **Property 3: Cache_Budget bounds without failing**
    - **Validates: Requirements 1.8**
    - `db` marker; `RuleBasedStateMachine` with `@invariant()`, Cache_Budget scaled to 256 KiB..4 MiB, including a monotonically-increasing-key rule that defeats LRU

- [x] 8. Node_Store write path
  - [x] 8.1 Implement the expansion write
    - `insert_expansion` as a single `INSERT ... ON CONFLICT (key_hi, key_lo) DO NOTHING RETURNING key_hi`, so node-plus-every-edge atomicity is a property of one row insert
    - Empty `RETURNING` means another task or another process won: return a duplicate-detected `WriteResult`, increment the duplicate counter, leave the retained row's evaluation fields untouched, terminate neither caller
    - `WriteResult` naming the Position_Key on failure, leaving no record from a failed write
    - SFEN comparison on read in the expansion path: a matching key with a differing stored SFEN creates no edge, changes nothing, and reports both SFEN strings
    - _Requirements: 1.6, 1.9, 3.5, 3.6, 10.3, 10.4, 11.6_

  - [x] 8.6 Implement the coalescing backup accumulator and flush
    - In-process accumulator keyed by Position_Key holding node visit delta, node value delta, `flags` OR mask, and a per-`move16` map of edge visit and value deltas; `backup()` is a plain `def` so no `await` sits between a descent's completion and its deltas becoming visible to `get`
    - Flusher coroutine on a 200 ms interval, on Cache_Budget pressure, and on stop; the dict snapshot-and-replace happens in one synchronous step before the first `await`
    - One `UPDATE` per node applying `visit_count + $`, `value_sum + $`, `flags | $`, and the packed-edge patch, issued as one `executemany` inside one transaction
    - The **documented client-side fallback** patch path, usable before the C extension exists: `SELECT edges ... FOR UPDATE`, patch with `packed_edge.patch_edges`, `UPDATE ... SET edges = $n`, all in one transaction, which gives the same no-lost-update guarantee at one extra round trip per node per flush
    - _Requirements: 4.4, 4.5, 8.7, 10.5, 11.8_

  - [x] 8.8 Implement propagation writes, stats, and cache release
    - `set_propagation` writing `prop_value`, `prop_best_move16`, `prop_epoch` through the accumulator; `book_meta.propagation_seq` / `propagation_done_seq` / `search_write_seq` maintenance
    - `stats()` exposing read and write latency histograms and cache byte accounting
    - RSS sampler at 10 s or shorter intervals driving LRU eviction until a later sample is within the Requirement 15.4 bound, retaining every written record and reporting a cache-release event
    - _Requirements: 9.1, 12.8, 15.4, 15.5_

  - [x] 8.10 Implement the in-flight claim mirror and startup clearing
    - Once per Report_Interval, upsert claims held longer than one Report_Interval into `in_flight_claim` with `process_id` and `worker_id`, and delete rows for released claims
    - At startup, `SELECT count(*)` for the reported cleared count then `TRUNCATE`, reading no `book_node` row and touching no visit count or value sum
    - _Requirements: 10.2, 11.3_

  - [ ]* 8.2 Write property test for expansion write atomicity
    - **Property 4: Expansion writes are atomic to concurrent readers**
    - **Validates: Requirements 1.6**
    - `db` marker

  - [ ]* 8.3 Write property test for transposition merging
    - **Property 11: Transposition merging reuses the existing node**
    - **Validates: Requirements 3.5**
    - `db` marker

  - [ ]* 8.4 Write property test for expansion idempotence and confluence
    - **Property 30: Expansion is idempotent and independent expansions are confluent**
    - **Validates: Requirements 10.3, 10.4, 11.5**
    - `db` marker; byte-identical row comparison, which is what the canonical zero encoding of absent Terashock fields exists for

  - [ ]* 8.5 Write property test for duplicate node creation
    - **Property 34: Duplicate node creation keeps the first**
    - **Validates: Requirements 11.6**
    - `db` marker; one parametrisation runs the two writers in **separate OS processes**, because that is the configuration the per-GPU process model produces and the one the per-process In_Flight_Set depends on

  - [ ]* 8.7 Write property test for backup arithmetic
    - **Property 14: Backup arithmetic and the perspective flip**
    - **Validates: Requirements 4.4**
    - `db` marker; assert both before and after `await store.flush()` so the accumulator read-through is part of the property

  - [ ]* 8.9 Write property test for cache release under memory pressure
    - **Property 47: Cache release under memory pressure is non-destructive**
    - **Validates: Requirements 15.5**
    - `db` marker; `@settings(deadline=None)`; RSS readings injected through a seam replacing the real sampler

  - [ ]* 8.11 Write property test for in-flight claim clearing
    - **Property 29: In_Flight_Set clearing is inert**
    - **Validates: Requirements 10.2**
    - `db` marker; assert every `book_node` row is byte-identical to its pre-startup state

- [ ] 9. `puct_edge` PostgreSQL C extension (optimisation over the working fallback)
  - [~] 9.1 Implement the extension
    - `dlshogi/book/pgext/puct_edge.c`: `puct_edge_backup(bytea, int2[], int4[], float8[]) RETURNS bytea`, `IMMUTABLE STRICT`, copying the input `bytea`, binary-searching the 20-byte records by `move16`, and adding the visit and value deltas in place
    - `dlshogi/book/pgext/Makefile` using PGXS, plus the control and SQL install files
    - _Requirements: 4.4, 11.8_

  - [~] 9.2 Wire backend selection into the Node_Store
    - Detect the extension at startup and select the in-database patch path, otherwise fall back to the documented `SELECT ... FOR UPDATE` client-side patch of task 8.6; log which path is in use
    - Both paths take PostgreSQL's row-level exclusive lock, so the no-lost-update guarantee holds either way
    - _Requirements: 4.4, 11.8_

  - [ ]* 9.3 Write property test for concurrent counter updates
    - **Property 33: Concurrent counter updates lose nothing**
    - **Validates: Requirements 11.8**
    - `db` marker; `@settings(deadline=None)`; parametrised over both the extension path and the client-side fallback, and deliberately bypassing the coalescing accumulator so that PostgreSQL's row lock is what is actually tested

- [ ] 10. Evaluator
  - [~] 10.1 Implement the batching Evaluator against an injectable session
    - `dlshogi/book/evaluator.py`: `EvalRequest` with an `asyncio.Future`, one collector coroutine per process over an `asyncio.Queue`, a preallocated staging buffer of `Batch_Size` entries filled by `cshogi.dlshogi.make_input_features`
    - Dispatch at `Batch_Size` or when `Batch_Timeout` has elapsed since the **earliest** pending request, with the earliest-arrival timestamp captured when the first request enters an empty buffer and `remaining` recomputed on each iteration
    - Policy decoding via `make_move_label` gathering legal-move logits out of the 2187-entry head, then a normalised softmax over exactly those entries; the degenerate policy-sum case substitutes `1/n` before the softmax and reports
    - Failure handling: an invocation error, a short result array, or a non-finite win rate or probability marks every request in the batch failed
    - The session is injected, so a stub returning drawn arrays satisfies the whole property suite with no GPU
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [~] 10.5 Wire the real onnxruntime session
    - TensorRT execution provider first with `trt_fp16_enable` and `trt_engine_cache_enable`, CUDA provider as the fallback, driven through `io_binding` with `bind_cpu_input("input1"/"input2")` and `bind_output("output_policy"/"output_value")`, following `dlshogi/utils/usi_policy_only.py`
    - Log the resolved provider at startup; zero-pad a short final batch rather than triggering an engine rebuild, discarding the padded outputs
    - _Requirements: 5.1, 5.2, 15.7_

  - [ ]* 10.2 Write property test for Evaluator outputs
    - **Property 17: Evaluator output is a normalised distribution and a win rate**
    - **Validates: Requirements 5.1, 5.4, 5.5, 5.7**
    - Stub session; `@settings(max_examples=1000)`

  - [ ]* 10.3 Write property test for Evaluator batching
    - **Property 18: Evaluator batching respects Batch_Size and Batch_Timeout**
    - **Validates: Requirements 5.2, 5.3, 15.7**
    - `RuleBasedStateMachine` over the virtual-clock loop with `arrive(n)` and `advance(ms)` rules and an `@invariant()` that no pending request has passed its deadline

  - [ ]* 10.4 Write property test for Evaluator failure handling
    - **Property 19: Evaluator failure leaves nothing behind**
    - **Validates: Requirements 5.6**
    - `db` marker, since the property asserts that no `book_node` row is created or changed

- [ ] 11. Repetition_Resolver and terminal states
  - [~] 11.1 Implement `dlshogi/book/repetition.py`
    - The resolver keeps **its own** `path_occurrences: dict[PositionKey, int]` keyed by the 128-bit Position_Key, incremented on push and decremented on pop, because cshogi's `is_draw()` was measured to return `REPETITION_DRAW` at the **second** occurrence and Requirement 8.1 needs the **fourth**
    - The resolver also keeps its own per-ply `board.is_check()` record, because cshogi exposes no `continuousCheck` equivalent; "checked at all four occurrences" is a scan of that record between the first and fourth occurrence indices
    - cshogi's `is_draw()` is consulted as a corroborating signal only, asserting agreement and reporting a discrepancy rather than driving the classification
    - The four-way truth table: neither side checking throughout and both sides checking throughout are draws valued by `Draw_Value_Black` / `Draw_Value_White` selected by the repeated Board_State's side to move; mover-only is a loss valued 0; opponent-only is a win valued 1
    - Terminal states: zero legal moves is loss for the side to move; `board.is_nyugyoku()` for the declaration win, plus an independently written WCSC predicate as the cross-check; terminal nodes get no edges and end the descent
    - Cyclic_Flag placement on the repeated node, the leaf, and every node between them, and only for repetition-derived values
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.9, 8.10, 8.11_

  - [ ]* 11.2 Write property test for the repetition truth table
    - **Property 23: Repetition classification truth table**
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.9, 8.10**
    - `@settings(max_examples=1000)`; includes the assertion that `is_draw()` is never used as the four-fold trigger

  - [ ]* 11.3 Write property test for terminal marking
    - **Property 24: Terminal marking and terminal nodes have no edges**
    - **Validates: Requirements 8.5, 8.6, 8.11**
    - The independent declaration-win predicate is written from the Requirement 8.6 text and compared against `board.is_nyugyoku()`; the generator straddles the 28/27 point boundary and the 10-piece boundary and includes in-check cases. If the property fails, the independent predicate becomes the implementation and cshogi's answer becomes the corroborating signal

  - [ ]* 11.4 Write perpetual-check example tests
    - `tests/book/test_repetition.py` example tests over real perpetual-check positions in `tests/book/fixtures/`, backing Property 23's truth table with positions random generation does not reliably produce
    - _Requirements: 8.3, 8.4_

- [ ] 12. Prior_Mixer and Terashock_Index
  - [~] 12.1 Implement `dlshogi/book/prior_mixer.py`
    - `t_raw(e) = exp(win_rate(e.ts_eval) / tau)` for edges carrying a Terashock evaluation and 0 otherwise, normalised to `t`, then `prior = (1 - w) * policy + w * t`
    - `win_rate(score) = 1 / (1 + exp(-score / Eval_Coef))`, the inverse of the `-log(1/wp - 1) * eval_coef` conversion used by `usi/UctSearch.cpp` and `make_book_minmax.py`
    - Illegal Terashock_Moves are dropped before `t_raw` is formed, so they influence neither the weights nor the edge set
    - Requirement 7.5's Terashock-derived initial mean value is a scoring-time `q0` derived from the stored `ts_eval`, not a stored `value_sum`, so the visit-count invariant stays exact
    - Mixing in float64, quantised to `prior_q16` on write
    - _Requirements: 7.3, 7.4, 7.5, 7.7, 7.9, 7.10_

  - [~] 12.3 Implement `dlshogi/book/terashock.py`
    - `import-terashock`: parse with Book_DB_Parser, `copy_records_to_table` into an unlogged staging table, then one `INSERT ... SELECT ... ON CONFLICT (key_hi, key_lo) DO UPDATE` so the **last** parsed entry wins and the conflict count is the duplicate-SFEN count
    - `moves` stored as `PACKED_TS_MOVE` records
    - Terashock_Index lookup by Position_Key with a small in-process LRU inside Cache_Budget; source identity (path, size, mtime, entry count, parser version) recorded in `book_meta` and compared at startup, triggering the import on mismatch or absence before any command is accepted
    - Unreadable or zero-entry Terashock_Book reports and exits before writing to the Node_Store
    - _Requirements: 7.1, 7.8_

  - [ ]* 12.2 Write property test for prior mixing
    - **Property 22: Prior mixing is a normalised convex combination**
    - **Validates: Requirements 7.3, 7.5, 7.7, 7.9, 7.10**
    - `@settings(max_examples=1000)`; the tolerance assertions are made against the `prior_q16`-decoded values so the quantisation error budget is part of the property

- [ ] 13. Search_Coordinator
  - [~] 13.1 Implement the In_Flight_Set and the claim reaper
    - `dlshogi/book/search.py`: `InFlightSet` backed by a plain `dict[PositionKey, Claim]` carrying worker id and claim timestamp, with no lock, because a single-threaded event loop cannot preempt between the test and the add
    - **Code invariant, not a language guarantee:** `test_and_add` is a plain `def`, contains no `await`, no `asyncio.sleep`, and no coroutine call between the membership test and the insertion. Record the invariant as a module docstring note; Property 31 is its executable statement
    - `discard`, the 1000 ms release contract, and the 300 s reaper coroutine waking once per second
    - _Requirements: 11.2, 11.3, 11.7_

  - [~] 13.3 Implement PUCT selection
    - Vectorized scoring over the decoded edge array: `q = where(n_eff > 0, w / max(n_eff, 1), q0)`, `score = q + c_puct * p * sqrt(n_par) / (1 + n_eff)`, with `n_eff = n + virtual_loss * in_flight_mask` and `w` left unchanged because virtual loss adds no wins
    - `w` needs no perspective flip, because a Book_Edge's accumulated value sum is stored from the parent Book_Node's side-to-move perspective
    - `q0` is 0.5 for an edge with no Terashock evaluation and the Terashock-derived win rate otherwise, computed as a vector over the `flags` bit-0 mask
    - Tie-break is `int(np.flatnonzero(score >= score.max() - 1e-6)[0])` over the USI-sorted array, **not** a bare `argmax`, so the 1e-6 tolerance resolves to the lexicographically smallest move USI
    - Excluded edges get `-inf`; virtual-loss terms are scoring-time only and are never written
    - _Requirements: 4.2, 4.7, 11.4_

  - [~] 13.6 Implement the descent task
    - Termination rules in order: `Max_Book_Ply` depth reached, terminal state, no edges (expand, and the newly expanded node is this descent's leaf), otherwise select an edge with in-flight children excluded for the rest of the descent; all edges excluded means abandon with no write, report, and start the next descent within 100 ms
    - None of the four rules reads the side to move, which is what makes colour-independence a property of the search rather than a carve-out
    - Root node creation when absent, at visit count 0 and value sum 0, before the first descent
    - Expansion: `test_and_add`, legal-move generation, Evaluator enqueue, Terashock lookup, Prior_Mixer, atomic `insert_expansion`, claim released in a `finally`
    - `asyncio.wait_for(self._descent(), timeout=10.0)` around the whole descent plus a per-node `loop.time() > deadline` check for CPU-bound stalls; the `finally` releases claims and discards the path without applying the backup
    - Value backup through the coalescing accumulator with the per-node and per-edge perspective conversion
    - _Requirements: 4.1, 4.3, 4.4, 4.6, 4.7, 4.8, 4.9, 4.11, 4.12, 5.6, 8.11, 11.4, 11.7, 15.6_

  - [~] 13.12 Implement the supervisor, resume, and shutdown
    - `TaskGroup`-style supervisor keeping exactly `Worker_Count` descent tasks alive with no bound on the number of descents, the wall-clock duration, or the Evaluator invocations per node; each completed task is replaced
    - Resume: a node with at least one Book_Edge or a non-empty terminal state counts as already evaluated and the Evaluator is not invoked for it; root-mismatch detection at startup
    - Stop: SIGINT / SIGTERM via `loop.add_signal_handler` plus the `stop` command sets a flag that starts no further descent and enqueues no further evaluation, lets in-flight descents finish, flushes the accumulator, and exits within 60 s; at the deadline the remaining tasks are cancelled, incomplete expansion writes discarded, and a forced-shutdown indication reported
    - `CancelledError` is caught only to run the `finally` and is then re-raised, never swallowed; every task is awaited by the supervisor so an exception becomes a Requirement 11.9 report rather than an unretrieved-task warning
    - Per-task `claimed` list released in a `finally`, so cancellation, timeout, and arbitrary exceptions all take the same path
    - _Requirements: 4.13, 10.1, 10.5, 10.6, 10.7, 10.8, 11.1, 11.9_

  - [ ]* 13.2 Write property test for the In_Flight_Set test-and-add
    - **Property 31: In_Flight_Set test-and-add admits exactly one claimant**
    - **Validates: Requirements 11.2, 11.7**
    - Suspension points are inserted immediately **before** and **after** the `test_and_add` call and never inside; the test passes for the synchronous implementation and fails for any implementation that awaits between the test and the add

  - [ ]* 13.4 Write property test for PUCT selection
    - **Property 12: PUCT selection maximises the score and breaks ties by USI order**
    - **Validates: Requirements 4.2, 4.7**
    - `@settings(max_examples=1000)`; the generator biases toward score differences in `(0, 1e-6)`, which is what separates the tolerance-aware selection from a bare `argmax`

  - [ ]* 13.5 Write property test for virtual loss
    - **Property 32: Virtual loss is applied in scoring and never persisted**
    - **Validates: Requirements 11.4**
    - `@settings(max_examples=500)` so all 17 Virtual_Loss values are seen many times

  - [ ]* 13.9 Write the colour-swap fixture and the colour-blindness property test
    - `tests/book/fixtures/colour_swap.py`: the purely textual SFEN transform (reverse rank and file order of the board field, invert piece-letter case, exchange `b`/`w`, swap the hand-field case groups) and the USI move transform (`f -> 10 - f`, `r -> 'j' - r`, `+` and drop prefixes untouched), with the involution check `swap(swap(x)) == x` asserted for every drawn SFEN and move plus `board.is_ok()` on every swapped position
    - The transform re-sorts each node's edges into ascending USI order and carries every per-edge field with its edge; the mock Evaluator is keyed by the **pre-swap** Position_Key
    - Both documented generator preconditions: reject positions whose side to move has its king inside the opponent's three ranks (the 28/27 declaration-win asymmetry), and draw priors and value sums so the PUCT argmax at every visited node is unique by more than the 1e-6 tolerance, asserting that margin as a generator postcondition
    - **Property 16: Descent is colour-blind**
    - **Validates: Requirements 4.11, 4.12**
    - `db` marker

  - [ ]* 13.10 Write property test for verbose move sequences and Max_Book_Ply
    - **Property 45: Verbose move sequences reconstruct the node**
    - **Validates: Requirements 4.6, 14.3**

  - [ ]* 13.11 Write property test for abandoned descents
    - **Property 46: Abandoned descents are inert**
    - **Validates: Requirements 4.9, 15.6**
    - Runs against the virtual-clock loop so the 10,000 ms deadline costs nothing to test

  - [ ]* 13.7 Write property test for expansion edge sets
    - **Property 13: Expansion writes exactly the legal move set**
    - **Validates: Requirements 4.3, 7.2, 7.4, 7.6**
    - `db` marker; the Terashock overlap generator reaches the all-illegal and empty cases

  - [ ]* 13.8 Write property test for the visit-count invariant
    - **Property 15: Visit-count invariant**
    - **Validates: Requirements 4.5**
    - `db` marker; `RuleBasedStateMachine` over a transposition-free synthetic tree with an `@invariant()` checked after **every** descent

  - [ ]* 13.13 Write property test for resume
    - **Property 28: Resume does not re-evaluate**
    - **Validates: Requirements 10.1**
    - `db` marker; the Evaluator stub calls `pytest.fail` if invoked for an excluded key

  - [ ]* 13.14 Write property test for abnormal task termination
    - **Property 35: Abnormal task termination releases only its own claims**
    - **Validates: Requirements 11.9**
    - `db` marker; termination kinds `cancel`, `raise`, and `timeout`, which are the three ways a descent task dies in this design

- [ ] 14. Value_Propagator
  - [ ]* 14.1 Write the independent negamax reference implementation
    - `tests/book/reference/negamax.py`: the Child_Contribution precedence recurrence written **from the Requirement 9 text alone**, importing nothing from `dlshogi.book`, reviewed against the requirements rather than against `propagate.py`
    - A straightforward recursive function, which is why the graph strategy caps at 2000 nodes
    - _Requirements: 9.2, 9.3, 9.4, 9.5, 9.9, 9.10, 9.12, 9.13, 9.15_

  - [~] 14.2 Implement `dlshogi/book/propagate.py`
    - Forward walk from the Root_Position over an **explicit** frame `list`, never Python recursion, bounded by `max(Max_Book_Ply, 1)` frames and by a hard 1024-frame guard when `Max_Book_Ply` is 0, counting cutoffs
    - Per frame: `set_sfen`, `np.frombuffer` decode, one batched `position_keys_after(sfen, edges["move16"])` call, a numpy threshold partition, then `asyncio.gather` of `get_many` for above-threshold children and `get_many_terminal_eval` for below-threshold children
    - Requirement 9 criterion 15's precedence order implemented **literally and in order**: terminal child; child on the current path (`Draw_Value_*` by the child's side to move, stored nowhere); below-threshold edge (visit count >= 1 gives `1 - clip(value_sum / visit_count, 0, 1)` with the clamp applied before the subtraction, visit count 0 gives the child's `eval_win_rate` else `Draw_Value_*`, neither descended into nor stored); otherwise the child's own propagated value
    - Memo in the database via `prop_epoch = pass_id` from `book_meta.propagation_seq`; Cyclic_Flag children are never memo hits, and only branch 4 consults the memo
    - `value = 1 - min(Child_Contributions)`, best move the first edge within 1e-6 of the minimum in the already-ascending USI order; write `prop_value`, `prop_best_move16`, `prop_epoch`
    - Missing root row reports and exits without modifying any row; `PropagationStats` reports nodes, draw revisits, unevaluated leaves, path cutoffs, below-threshold edges, and the root value; `propagation_done_seq` set on completion
    - Resident memory stays within Cache_Budget by construction: frame stack, path key set, prefetch buffer, and an LRU sized to what remains
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10, 9.11, 9.12, 9.13, 9.14, 9.15, 9.16_

  - [ ]* 14.3 Write property test for the negamax recurrence
    - **Property 26: Propagation satisfies the negamax recurrence over Child_Contributions**
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.9, 9.10, 9.12, 9.13, 9.15**
    - `db` marker; `@settings(max_examples=1000)`; compared against `tests/book/reference/negamax.py`; the strategy draws the threshold first and then visit counts straddling and sitting exactly on it, and draws value sums outside `[0, visit_count]` so the clamp is exercised

  - [ ]* 14.4 Write property test for propagation idempotence
    - **Property 27: Propagation is idempotent**
    - **Validates: Requirements 9.8**
    - `db` marker; at least half the examples use a non-zero threshold, and `@example` pins a Cyclic_Flag node reachable only through below-threshold edges

  - [ ]* 14.5 Write property test for the threshold extremes
    - **Property 48: The visit threshold is inert at 0 and total above every visit count**
    - **Validates: Requirements 9.14, 9.16**
    - `db` marker; frame counts and the set of written Book_Nodes are observed through **counting seams** on the frame stack and on `set_propagation`, not inferred from values, because the property is about the absence of recursion and of writes; the third conjunct compares the packed edge blobs and `eval_win_rate` byte-for-byte before and after the pass

  - [ ]* 14.6 Write property test for Cyclic_Flag placement and non-reuse
    - **Property 25: Cyclic_Flag placement and non-reuse of cyclic values**
    - **Validates: Requirements 8.7, 8.8**
    - `db` marker; the third conjunct runs a propagation pass over graphs whose Cyclic_Flag nodes carry poisoned stored values

- [ ] 15. Book_Exporter
  - [~] 15.1 Implement export phase 1 emission and filters
    - `dlshogi/book/export.py`: an asyncpg server-side cursor streaming `book_node` in heap order with **no predicate at all**, so both sides to move are covered and the only exclusions are the per-edge ones
    - Per-edge exclusions: visit-count ratio below `Export_Visit_Threshold`; parent visit count 0; child with no `prop_value`; child whose `prop_value` is stale, tested as `prop_value IS NULL OR prop_epoch <> propagation_done_seq` over a batched child lookup that fetches `prop_epoch` alongside `prop_value`
    - Legality assertion via `board.move_from_move16` against `board.legal_moves`, aborting the export on failure since a violation means the graph is corrupt
    - Record emission into a preallocated `np.empty(N, dtype=cshogi.BookEntry)` 64 MiB buffer with `key = node.apery_key`, `fromToPro = move16`, `count = clip(visit_count, 0, 65535)`, `score = clip(round(-log(1/v - 1) * Eval_Coef), INT32_MIN, INT32_MAX)` where `v = 1 - child.prop_value` in the parent's perspective
    - `ExportCounts` reporting records written, entries written, and excluded edges as the criterion 5 count plus the criterion 9 count; the stale-propagation warning when `search_write_seq > propagation_done_seq` or `propagation_done_seq = 0`, emitted before the first record with the export continuing
    - _Requirements: 12.1, 12.5, 12.6, 12.7, 12.8, 12.9, 12.11, 12.12_

  - [~] 15.4 Implement the external sort and the Apery writer
    - In-buffer ordering by `np.lexsort` over `(fromToPro asc, -count, -score, key)` applied last-key-first, with `score` and `count` widened to `int64` before negation and `key` read through the `<u8` field so the primary comparison is unsigned
    - Run files written with `arr[:n].tofile(...)`, phase 2 k-way merge over `np.memmap` windows with `heapq.merge` on the same total order
    - Output written to a temporary path in the destination directory and `os.replace()`-ed on success, so an unopenable or failing path leaves nothing at the target
    - _Requirements: 12.2, 12.3, 12.10_

  - [~] 15.6 Implement the YaneuraOu `.db` export path
    - Phase 1 emits length-prefixed `(sfen, move-line block)` records, sorted by `bytes` SFEN keys, which is Python's native byte-wise order and exactly what Requirement 12.4 asks for
    - Phase 2 merges with `heapq.merge` and streams through Book_DB_Printer; the entry count is known before phase 2 opens the output so the `# NOE:` line needs no rewind; Terashock_Moves within an entry are ordered descending by evaluation value
    - _Requirements: 12.4, 12.10, 12.11, 12.12_

  - [ ]* 15.2 Write property test for export record mapping
    - **Property 36: Export record mapping**
    - **Validates: Requirements 12.1, 12.6**
    - `db` marker; `@settings(max_examples=1000)`

  - [ ]* 15.3 Write property test for the export filter
    - **Property 38: Export filter partitions the edge set**
    - **Validates: Requirements 12.5, 12.9, 12.11, 12.12**
    - `db` marker; the written-plus-excluded total must equal the whole Book_Edge count, which is the conjunct that catches a stray side-to-move predicate

  - [ ]* 15.5 Write property test for Apery file ordering
    - **Property 37: Apery file ordering is total and content-determined**
    - **Validates: Requirements 12.2, 12.3**
    - `db` marker; `@settings(max_examples=1000)`; the `apery_key` generator deliberately straddles 2^63 so the signed/unsigned trap in the `bigint` column and the `np.lexsort` key construction are both exercised

  - [ ]* 15.7 Write property test for YaneuraOu export ordering
    - **Property 40: YaneuraOu export ordering**
    - **Validates: Requirements 12.4**
    - `db` marker

  - [ ]* 15.8 Write property test for exported move legality
    - **Property 39: Exported moves are legal**
    - **Validates: Requirements 12.7**
    - `db` marker; verification against a freshly re-read file via `np.fromfile(path, cshogi.BookEntry)` and `board.move_from_move16`, not against in-memory state

- [ ] 16. Progress_Reporter
  - [~] 16.1 Implement `dlshogi/book/report.py`
    - stdlib `logging` with a `RotatingFileHandler` plus a stderr handler, one structured JSON record per `Report_Interval`, woken by `loop.call_later` on the interval boundary so the `max(1 s, 0.1 * Report_Interval)` deadline is met by construction
    - Record contents: elapsed run time, Book_Node and Book_Edge counts from incrementally maintained counters rather than `SELECT count(*)`, cumulative completed descents, descents per second and Evaluator batches per second over the most recent interval, mean and p95 read and write latency from per-interval numpy bucket histograms reset each interval
    - Cumulative counters for Terashock injections, illegal Terashock discards, Evaluator failures, and duplicate node creations, as plain `int` attributes on the single event loop
    - The degraded-throughput and throughput-recovered state machine, including intervals whose measured rate is 0, at most one warning per Throughput_Grace_Period, and a restart of the continuous-duration measurement on recovery
    - Verbose mode logging the root-to-node USI move sequence for each new Book_Node; an unwritable log destination reported to the Operator at most once per interval with the record discarded and the run continuing
    - A `--merge` mode combining per-process log files
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7_

  - [ ]* 16.2 Write property test for progress counters
    - **Property 43: Progress counters are running sums**
    - **Validates: Requirements 14.2, 14.6**
    - `RuleBasedStateMachine` with `event(category)` and `tick()` rules over the virtual clock, sequences reaching 10,000 events via `stateful_step_count`

  - [ ]* 16.3 Write property test for the throughput warning state machine
    - **Property 44: Throughput warning state machine**
    - **Validates: Requirements 14.4, 14.5**
    - `RuleBasedStateMachine`; `@settings(max_examples=1000)`; sequences reaching 2000 intervals, with the grace period expressed as an integral number of intervals in half the cases

- [ ] 17. CLI wiring, startup flow, and example tests
  - [~] 17.1 Wire `dlshogi/book/__main__.py` and the startup flow
    - `argparse` subcommands `search`, `propagate`, `export`, `import-terashock` dispatching to the implemented components
    - The startup order of the design's Error Handling flow: load and validate configuration, log it with credentials redacted, import `dlshogi.cppshogi`, connect with the retry schedule, create or repair the schema, check version and Zobrist fingerprint, check the Root_Position, import or verify the Terashock index, count and truncate `in_flight_claim`, then accept commands
    - `search` spawns one OS process per GPU, each repeating the read-only half of the checks and exiting if any disagrees; only the parent creates or repairs the schema; the parent forwards stop signals to every child inside one 60 s budget
    - Reject `import-terashock` when no Terashock path is configured, and reject any command under an invalid configuration
    - _Requirements: 4.10, 4.13, 10.5, 10.6, 11.1, 13.1, 13.4, 13.5, 13.8, 13.9_

  - [ ]* 17.2 Write the example tests for the criteria the design classified as EXAMPLE
    - `tests/book/test_units.py`: the three-way `get` result distinction; a fault-injected expansion write failure; connection-failure and schema-creation-failure reports and exits; Root_Position recording and read-back; a forged Position_Key collision row; descent start at the root at depth 0; root node creation when absent; the 100 ms restart after an abandoned descent; the three unreadable Terashock-input cases; the propagation report; propagate with no root row; stop within 60 s, no work while stopping, forced shutdown, and root mismatch; the three propagation-freshness states of the stale-propagation warning; the two unopenable-output-path cases; absent Terashock path and rejected `import-terashock`; the unwritable log destination
    - _Requirements: 1.4, 1.9, 2.3, 2.6, 2.7, 3.6, 4.1, 4.8, 4.9, 7.8, 9.7, 9.11, 10.5, 10.6, 10.7, 10.8, 12.8, 12.10, 13.8, 13.9, 14.7_

  - [ ]* 17.3 Write the smoke tests
    - `tests/book/test_smoke.py`: the schema exists after startup; `assert POSITION_KEY.itemsize == 16`; every Book_Node and Book_Edge write of a run lands in the one Book_Graph named by the configured connection settings and in no other; `Worker_Count` descent tasks are created against one store; every configuration name and every command name is recognised
    - _Requirements: 2.1, 2.4, 3.1, 4.10, 11.1, 13.1, 13.4_

- [~] 18. Checkpoint - full property suite green
  - Ensure all tests pass, ask the user if questions arise.
  - `pytest -m "not db"` and the full suite including `db` both pass; all 48 properties are implemented, each by exactly one property-based test

- [ ] 19. Integration and scale verification
  - [ ]* 19.1 Write the scale, latency, and structural integration test
    - Extend `tests/book/test_integration.py`: generate a synthetic 10^8-Book_Node graph with an asyncpg `copy_records_to_table` loader using the exact packed encoding, then run a real search for at least 300 s of warmup plus a measurement window
    - Sample 10,000 consecutive node-plus-edges reads for the 20 ms p95 / 200 ms p99 bound, 10,000 consecutive descents for the 1000 ms p95 / 10,000 ms max bound, 60-second windows for the descent rate and the mean Evaluator batch size, and RSS every 10 s for the memory bound
    - The three structural assertions that validate the storage design rather than the code: `pg_column_size(edges) <= 2000` sampling (rows stay inline, not TOASTed), `pg_stat_user_tables.n_tup_hot_upd / n_tup_upd > 0.95` (backups stay HOT), and the measured per-process descent rate falling inside the 600-1,250/s band the Performance budget predicts. A measured rate outside that band means the design document's model is wrong and the document is what gets fixed
    - Assert the search session's effective `synchronous_commit = off` and that `book_node` is a logged relation
    - _Requirements: 1.5, 15.1, 15.2, 15.3, 15.4, 15.7_

  - [ ]* 19.2 Write the crash-recovery and connection-loss integration tests
    - `SIGKILL` at randomized points during expansion writes and during accumulator flushes, restart, and assert every Position_Key is either absent or complete and that the run resumes
    - Terminate backends with `pg_terminate_backend` during a search and assert suspension, reconnection on the schedule, and resumption
    - _Requirements: 2.9, 9.6, 10.3, 10.7_

  - [ ]* 19.3 Write the multi-process integration test
    - Two search processes against one database with overlapping search regions; assert exactly one row per Position_Key, a duplicate count consistent with the observed overlap, and no lost counter updates across the two processes
    - Also assert the Terashock_Index 1 ms p95 lookup bound over an index of at least 10^7 entries
    - _Requirements: 7.1, 11.1, 11.6_

  - [ ]* 19.4 Write the end-to-end book interoperability test
    - A small real search, a propagation pass, and an export; then read the resulting `book.bin` back through the existing `dlshogi/utils/book.py` and through cshogi's `BookEntry` dtype and `Board.book_key()`, confirming an independent reader agrees on keys, moves, counts, and scores
    - This is the only check that proves the exported file works with the game-time engine, which is the whole point of the export
    - Also export and re-parse the YaneuraOu `.db` output through `tests/book/reference/book_db.py`
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.6, 12.7, 12.11, 12.12_

- [~] 20. Final checkpoint
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP. Skipping them skips the
  property suite, which is where nearly all of the correctness evidence for this feature lives, so
  the recommendation is to skip none of them.
- Task 2 is a hard prerequisite: nothing else can be built or tested against real positions until the
  128-bit key binding lands, and Properties 8 and 10 plus the task 2.3 gate test are what declare it
  complete. The 64-bit `book_key()` fallback the design records is a documented deviation, not a
  compliant path, and is not implemented by these tasks.
- Task 9 is an optimisation. Task 8.6 delivers the documented `SELECT ... FOR UPDATE` client-side
  patch first, so the search is usable before the extension exists, and Property 33 validates both
  paths.
- The two reference implementations under `tests/book/reference/` are written from the requirement
  text alone and import nothing from `dlshogi.book`. Both sides are Python now, so that independence
  is deliberate rather than a consequence of a language boundary; tasks 5.1 and 14.1 state it
  explicitly.
- Every property task names the property number and title verbatim from the design and the
  requirements clauses it discharges. No property is collapsed, split, or renumbered.
- Properties 1-5, 11, 13-16, 19, 28-40, and 47 carry the `db` marker so `pytest -m "not db"` stays
  fast. The GPU is never required by the property suite; only task 19 needs one.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "2.1"] },
    { "id": 1, "tasks": ["2.2"] },
    { "id": 2, "tasks": ["2.3", "3.1", "3.5", "4.1", "5.1", "7.1", "14.1"] },
    { "id": 3, "tasks": ["3.6", "5.2", "7.2", "16.1"] },
    { "id": 4, "tasks": ["3.7", "7.6", "10.1", "11.1", "12.1"] },
    { "id": 5, "tasks": ["3.2", "4.2", "5.3", "7.3", "8.1", "10.2", "11.2", "12.2", "12.3", "16.2"] },
    { "id": 6, "tasks": ["3.3", "4.3", "5.4", "7.4", "10.3", "11.3", "13.1", "16.3"] },
    { "id": 7, "tasks": ["3.4", "7.5", "8.6", "10.4", "11.4", "13.2"] },
    { "id": 8, "tasks": ["7.7", "9.1", "10.5", "13.3"] },
    { "id": 9, "tasks": ["7.8", "9.2", "13.4"] },
    { "id": 10, "tasks": ["7.9", "8.8", "13.5"] },
    { "id": 11, "tasks": ["8.2", "8.10", "13.6"] },
    { "id": 12, "tasks": ["8.3", "14.2"] },
    { "id": 13, "tasks": ["8.4", "13.9", "15.1"] },
    { "id": 14, "tasks": ["8.5", "13.10", "14.3", "15.4"] },
    { "id": 15, "tasks": ["8.7", "13.11", "14.4", "15.2", "15.6"] },
    { "id": 16, "tasks": ["9.3", "13.12", "14.5", "15.3"] },
    { "id": 17, "tasks": ["8.9", "14.6", "15.5", "17.1"] },
    { "id": 18, "tasks": ["8.11", "15.7", "17.2", "17.3"] },
    { "id": 19, "tasks": ["13.7", "15.8"] },
    { "id": 20, "tasks": ["13.8", "19.1"] },
    { "id": 21, "tasks": ["13.13", "19.2"] },
    { "id": 22, "tasks": ["13.14", "19.3"] },
    { "id": 23, "tasks": ["19.4"] }
  ]
}
```

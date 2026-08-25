-- PUCT Book Builder schema.
--
-- Requirements: 1.1, 1.2, 2.4, 2.7
--
-- This file is applied verbatim by Node_Store schema creation (design.md,
-- "Error Handling" -> startup flow node C7) and is also the reference schema
-- Property 5 (schema repair) diffs against a pg_catalog snapshot.
--
-- Layout summary (see design.md "Schema" and "The backup write path"):
--   * book_meta          -- one row, schema version / Zobrist fingerprint / Root_Position / seq counters
--   * book_node          -- one row per Book_Node, edges packed into a single bytea column
--   * terashock_entry    -- Position_Key -> Terashock_Move list, imported from the .db file
--   * in_flight_claim    -- diagnostic mirror of the per-process In_Flight_Set (Requirement 10.2)

CREATE TABLE book_meta (
    id                     smallint PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    schema_version         text     NOT NULL,
    zobrist_fingerprint    bigint   NOT NULL,
    root_sfen              text     NOT NULL,
    root_key_hi            bigint   NOT NULL,
    root_key_lo            bigint   NOT NULL,
    search_write_seq       bigint   NOT NULL DEFAULT 0,
    propagation_seq        bigint   NOT NULL DEFAULT 0,
    propagation_done_seq   bigint   NOT NULL DEFAULT 0,
    terashock_source_path  text,
    terashock_source_size  bigint,
    terashock_source_mtime bigint,
    terashock_entry_count  bigint,
    created_at             timestamptz NOT NULL DEFAULT now()
);

-- book_node carries exactly one index: the primary key on (key_hi, key_lo).
-- Every column a backup or a propagation write touches -- visit_count,
-- value_sum, flags, edges, prop_value, prop_best_move16, prop_epoch -- is
-- unindexed. That is the precondition for HOT updates (see design.md, "The
-- storage layout decision" and "The backup write path"), and it is why no
-- index is created on apery_key (export scans the heap sequentially and
-- sorts externally), on prop_epoch, or on any child key: a packed edge list
-- admits no index on child Position_Key at all, since the child key is
-- materialised on demand rather than stored (Requirement 1.2).
--
-- book_node is deliberately NOT UNLOGGED. UNLOGGED relations are truncated
-- by crash recovery, which would destroy the entire graph -- not merely the
-- in-flight writes -- directly contradicting Requirements 2.3, 10.1, and
-- 10.3, which all presuppose that completed expansion writes survive
-- termination. Resumability makes the graph reconstructible by more search,
-- not disposable. (What resumability does justify is
-- `synchronous_commit = off` on the search session, set at the connection
-- level by Node_Store rather than in this schema file; WAL is still written
-- and crash recovery is still atomic per transaction, so completed writes
-- are never lost outright, only the last `wal_writer_delay` of them at most.)
CREATE TABLE book_node (
    key_hi           bigint  NOT NULL,   -- Position::getBoardKey(); bit 0 = side to move
    key_lo           bigint  NOT NULL,   -- Position::getHandKey()
    sfen             text    NOT NULL,   -- board.sfen(), <= 128 chars (Req 1.1)
    apery_key        bigint  NOT NULL,   -- board.book_key()            (Req 12.1)
    visit_count      bigint  NOT NULL DEFAULT 0 CHECK (visit_count >= 0),
    value_sum        float8  NOT NULL DEFAULT 0,
    terminal         smallint NOT NULL DEFAULT 0,  -- 0 none, 1 win for stm, 2 loss for stm
    flags            smallint NOT NULL DEFAULT 0,  -- bit0 Cyclic_Flag
    eval_win_rate    real,              -- NULL until the Evaluator has run (Req 9.9, 9.10)
    prop_value       float8,            -- NULL until propagated          (Req 9.1)
    prop_best_move16 smallint,          -- NULL when absent               (Req 9.3, 9.4)
    prop_epoch       integer NOT NULL DEFAULT 0,   -- propagation memo tag
    edge_count       smallint NOT NULL DEFAULT 0,
    edges            bytea   NOT NULL DEFAULT '',  -- edge_count * PACKED_EDGE
    CONSTRAINT book_node_pkey PRIMARY KEY (key_hi, key_lo),
    CONSTRAINT book_node_edges_len CHECK (octet_length(edges) = edge_count * 20)
) WITH (fillfactor = 70);

-- Keep the varlena columns inline (MAIN) rather than allowing PostgreSQL to
-- move them out of line to TOAST as a first resort. MAIN still compresses
-- but treats out-of-line storage as a last resort, used only when the tuple
-- cannot fit a page at all -- see design.md, "Inline versus TOAST, and total
-- size". Storing the child Position_Key instead of deriving it would have
-- pushed the typical row past the 2032-byte TOAST threshold; that is why it
-- is not stored (Requirement 1.2).
ALTER TABLE book_node ALTER COLUMN edges SET STORAGE MAIN;
ALTER TABLE book_node ALTER COLUMN sfen  SET STORAGE MAIN;

CREATE TABLE terashock_entry (
    key_hi bigint NOT NULL,
    key_lo bigint NOT NULL,
    sfen   text   NOT NULL,
    moves  bytea  NOT NULL,               -- packed Terashock_Move records, 11 bytes each
    PRIMARY KEY (key_hi, key_lo)
) WITH (fillfactor = 100);

-- in_flight_claim is a diagnostic mirror of the per-process, in-memory
-- In_Flight_Set (design.md, "In_Flight_Set"). It is a regular logged table,
-- not UNLOGGED, precisely because UNLOGGED relations are truncated by crash
-- recovery, which would make the startup-reported cleared count always 0
-- and defeat Requirement 10.2's diagnostic purpose entirely.
CREATE TABLE in_flight_claim (
    key_hi     bigint NOT NULL,
    key_lo     bigint NOT NULL,
    process_id integer NOT NULL,
    worker_id  integer NOT NULL,
    claimed_at timestamptz NOT NULL,
    PRIMARY KEY (key_hi, key_lo)
);

-- puct_edge extension, version 1.0
--
-- puct_edge_backup(bytea, int2[], int4[], float8[]) RETURNS bytea
--
-- See dlshogi/book/pgext/puct_edge.c for the full contract and
-- design.md's "The backup write path" section for the call site.

CREATE FUNCTION puct_edge_backup(bytea, int2[], int4[], float8[])
RETURNS bytea
AS '$libdir/puct_edge', 'puct_edge_backup'
LANGUAGE C IMMUTABLE STRICT;

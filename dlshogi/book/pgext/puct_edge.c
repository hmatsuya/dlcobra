/*
 * puct_edge.c -- in-database packed-edge backup patch.
 *
 * puct_edge_backup(bytea, int2[], int4[], float8[]) RETURNS bytea
 * IMMUTABLE STRICT
 *
 * This is the C extension referenced by design.md's "The backup write
 * path" section, which shows the call site as part of a single UPDATE:
 *
 *   UPDATE book_node
 *      SET visit_count = visit_count + $3,
 *          value_sum   = value_sum   + $4,
 *          flags       = flags | $5,
 *          edges       = puct_edge_backup(edges, $6::int2[], $7::int4[], $8::float8[])
 *    WHERE key_hi = $1 AND key_lo = $2;
 *
 * It discharges Requirements 4.4 (backup increments) and 11.8 (no lost
 * update): running the patch inside the UPDATE, under PostgreSQL's
 * row-level exclusive lock, is what gives concurrent backups to the same
 * Book_Node's packed edge list the same no-lost-update guarantee the
 * client-side fallback (`dlshogi/book/packed_edge.py`'s `patch_edges`,
 * used by `node_store.py`'s `_apply_backup_row` until task 9.2 wires this
 * extension in) gets from `SELECT ... FOR UPDATE`.
 *
 * Contract:
 *   - Copies the input bytea; the input is never mutated in place
 *     (Postgres bytea arguments may point at cached/shared storage, so
 *     mutating in place would corrupt unrelated readers).
 *   - The three arrays (move16 int2[], visit deltas int4[], value deltas
 *     float8[]) must have equal length and no NULL elements; STRICT means
 *     Postgres will not call this function at all if one of the four
 *     *arguments* is SQL NULL, but array *elements* can independently be
 *     NULL even when the array itself is non-NULL, so that is checked
 *     here.
 *   - Every move16 named in the arrays must already have a record in the
 *     blob; a name with no matching record is an error (mirrors
 *     packed_edge.patch_edges, which raises KeyError in that case).
 *   - Zero-length arrays are legal and return an unmodified copy.
 *
 * On-disk record layout (dlshogi/book/packed_edge.py's PACKED_EDGE
 * dtype), 20 bytes, little-endian, no padding:
 *
 *   offset  0: move16       u2
 *   offset  2: prior_q16    u2
 *   offset  4: ts_depth     u1
 *   offset  5: flags        u1
 *   offset  6: ts_eval      i2
 *   offset  8: visit_count  u4   <- patched here (wraparound add)
 *   offset 12: value_sum    f8   <- patched here (float64 add)
 *
 * design.md's prose says this function "binary-searches" the records by
 * move16. That is not correct for the actual on-disk order: the blob is
 * sorted ascending by each move's USI notation (see packed_edge.py's
 * encode_edges docstring), not by the numeric move16 value, because
 * promotion sets bit 14 of move16 (Apery's proFromAndTo) and that bit
 * does not track USI lexicographic order. A binary search keyed on
 * move16 over a USI-sorted array would silently return wrong records
 * whenever a promotion is present. This function therefore does a
 * linear scan by move16 for each requested edge, exactly like the
 * client-side fallback (packed_edge.patch_edges) does with its
 * move16 -> index dict. A linear scan over the 1-600 records design.md
 * expects per node (see "Performance budget") costs nothing that
 * matters; the wire format's USI ordering is not touched by this
 * function and must not be re-sorted.
 *
 * The codebase already assumes a little-endian host throughout (see
 * packed_edge.py's numpy dtype, built from "<u2"/"<i2"/"<u4"/"<f8"), so
 * reading/writing these fields via memcpy into properly-aligned local
 * variables, with no byte-swapping, is consistent with the rest of the
 * repository and is what this file does.
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/array.h"
#include "utils/lsyscache.h"
#include "catalog/pg_type.h"
#include <string.h>

PG_MODULE_MAGIC;

#define PACKED_EDGE_SIZE 20
#define OFFSET_MOVE16      0
#define OFFSET_VISIT_COUNT 8
#define OFFSET_VALUE_SUM  12

/*
 * ---------------------------------------------------------------------
 * PostgreSQL-independent core.
 *
 * This section operates only on raw unsigned char buffers and plain C
 * types -- no Datum, no fmgr.h, no palloc -- so it can be compiled and
 * exercised by a standalone test harness with a plain C compiler and no
 * PostgreSQL server headers.
 * ---------------------------------------------------------------------
 */

/*
 * Read the little-endian uint16 move16 field of the record starting at
 * `record`.
 */
static uint16_t
puct_edge_read_move16(const unsigned char *record)
{
    uint16_t value;
    memcpy(&value, record + OFFSET_MOVE16, sizeof(value));
    return value;
}

/*
 * Add `visit_delta` to the little-endian uint32 visit_count field of the
 * record starting at `record`. Wraparound is intentional and matches the
 * numpy uint32 add semantics the Python fallback relies on.
 */
static void
puct_edge_add_visit_delta(unsigned char *record, int32_t visit_delta)
{
    uint32_t visit_count;

    memcpy(&visit_count, record + OFFSET_VISIT_COUNT, sizeof(visit_count));
    visit_count = (uint32_t) (visit_count + (uint32_t) visit_delta);
    memcpy(record + OFFSET_VISIT_COUNT, &visit_count, sizeof(visit_count));
}

/*
 * Add `value_delta` to the little-endian float64 value_sum field of the
 * record starting at `record`.
 */
static void
puct_edge_add_value_delta(unsigned char *record, double value_delta)
{
    double value_sum;

    memcpy(&value_sum, record + OFFSET_VALUE_SUM, sizeof(value_sum));
    value_sum += value_delta;
    memcpy(record + OFFSET_VALUE_SUM, &value_sum, sizeof(value_sum));
}

/*
 * Apply one (move16, visit_delta, value_delta) patch to `buf`, which holds
 * `buf_len` bytes of consecutive PACKED_EDGE_SIZE-byte records. Finds the
 * matching record by a linear scan over move16 (see the file header
 * comment for why this must not be a binary search) and adds the deltas
 * in place at the visit_count and value_sum offsets, leaving every other
 * field of every record untouched.
 *
 * Returns true if a matching record was found and patched, false if no
 * record in `buf` has move16 == `move16`. `buf_len` is assumed to already
 * be a multiple of PACKED_EDGE_SIZE; the caller validates that.
 */
static bool
puct_edge_patch_one(unsigned char *buf, size_t buf_len, uint16_t move16,
                     int32_t visit_delta, double value_delta)
{
    size_t offset;

    for (offset = 0; offset < buf_len; offset += PACKED_EDGE_SIZE)
    {
        unsigned char *record = buf + offset;

        if (puct_edge_read_move16(record) == move16)
        {
            puct_edge_add_visit_delta(record, visit_delta);
            puct_edge_add_value_delta(record, value_delta);
            return true;
        }
    }
    return false;
}

/*
 * ---------------------------------------------------------------------
 * PostgreSQL wrapper.
 * ---------------------------------------------------------------------
 */

PG_FUNCTION_INFO_V1(puct_edge_backup);

Datum
puct_edge_backup(PG_FUNCTION_ARGS)
{
    bytea      *in_blob = PG_GETARG_BYTEA_PP(0);
    ArrayType  *move16_array = PG_GETARG_ARRAYTYPE_P(1);
    ArrayType  *visit_delta_array = PG_GETARG_ARRAYTYPE_P(2);
    ArrayType  *value_delta_array = PG_GETARG_ARRAYTYPE_P(3);

    int         in_len;
    bytea      *result;
    unsigned char *out_data;

    Datum      *move16_elems;
    bool       *move16_nulls;
    int         move16_count;
    Datum      *visit_delta_elems;
    bool       *visit_delta_nulls;
    int         visit_delta_count;
    Datum      *value_delta_elems;
    bool       *value_delta_nulls;
    int         value_delta_count;

    int16       elmlen16;
    bool        elmbyval16;
    char        elmalign16;
    int16       elmlen32;
    bool        elmbyval32;
    char        elmalign32;
    int16       elmlen64;
    bool        elmbyval64;
    char        elmalign64;

    int         i;

    in_len = VARSIZE_ANY_EXHDR(in_blob);

    if (in_len % PACKED_EDGE_SIZE != 0)
        ereport(ERROR,
                (errcode(ERRCODE_DATA_CORRUPTED),
                 errmsg("puct_edge_backup: corrupt packed edge blob "
                        "(length %d is not a multiple of %d)",
                        in_len, PACKED_EDGE_SIZE)));

    get_typlenbyvalalign(INT2OID, &elmlen16, &elmbyval16, &elmalign16);
    deconstruct_array(move16_array, INT2OID, elmlen16, elmbyval16, elmalign16,
                       &move16_elems, &move16_nulls, &move16_count);

    get_typlenbyvalalign(INT4OID, &elmlen32, &elmbyval32, &elmalign32);
    deconstruct_array(visit_delta_array, INT4OID, elmlen32, elmbyval32, elmalign32,
                       &visit_delta_elems, &visit_delta_nulls, &visit_delta_count);

    get_typlenbyvalalign(FLOAT8OID, &elmlen64, &elmbyval64, &elmalign64);
    deconstruct_array(value_delta_array, FLOAT8OID, elmlen64, elmbyval64, elmalign64,
                       &value_delta_elems, &value_delta_nulls, &value_delta_count);

    if (move16_count != visit_delta_count || move16_count != value_delta_count)
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("puct_edge_backup: move16, visit delta, and value "
                        "delta arrays must have equal length "
                        "(got %d, %d, %d)",
                        move16_count, visit_delta_count, value_delta_count)));

    for (i = 0; i < move16_count; i++)
    {
        if (move16_nulls[i] || visit_delta_nulls[i] || value_delta_nulls[i])
            ereport(ERROR,
                    (errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
                     errmsg("puct_edge_backup: array element %d is NULL",
                            i + 1)));
    }

    /*
     * Copy the input bytea (including its varlena header) so the
     * original, which Postgres may have handed us a pointer into shared
     * or cached storage for, is never mutated in place.
     */
    result = (bytea *) palloc(VARSIZE_ANY(in_blob));
    memcpy(result, in_blob, VARSIZE_ANY(in_blob));
    out_data = (unsigned char *) VARDATA_ANY(result);

    for (i = 0; i < move16_count; i++)
    {
        uint16_t move16 = (uint16_t) DatumGetInt16(move16_elems[i]);
        int32_t visit_delta = DatumGetInt32(visit_delta_elems[i]);
        double value_delta = DatumGetFloat8(value_delta_elems[i]);

        if (!puct_edge_patch_one(out_data, (size_t) in_len, move16,
                                  visit_delta, value_delta))
            ereport(ERROR,
                    (errcode(ERRCODE_NO_DATA_FOUND),
                     errmsg("puct_edge_backup: move16 %u not present in "
                            "packed edge blob", (unsigned int) move16)));
    }

    pfree(move16_elems);
    pfree(move16_nulls);
    pfree(visit_delta_elems);
    pfree(visit_delta_nulls);
    pfree(value_delta_elems);
    pfree(value_delta_nulls);

    PG_RETURN_BYTEA_P(result);
}

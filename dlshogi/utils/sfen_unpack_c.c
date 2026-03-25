#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

/*
 * Fast C implementation of YaneuraOu packed SFEN → SFEN string conversion.
 * Replaces the pure Python sfen_converter.unpack() for ~100x speedup.
 */

/* Piece characters: index 0=unused, 1=pawn..7=rook, 8=king */
static const char piece_upper[] = " PLNSGBR";
static const char piece_lower[] = " plnsgbr";

/* Read n bits from packed data (LSB first within each byte) */
static inline uint32_t read_bits(const uint8_t *data, int *bit_pos, int n) {
    uint32_t val = 0;
    for (int i = 0; i < n; i++) {
        int byte_idx = *bit_pos >> 3;
        int bit_idx = *bit_pos & 7;
        val |= ((data[byte_idx] >> bit_idx) & 1) << i;
        (*bit_pos)++;
    }
    return val;
}

/* Read a single bit from the stream */
static inline int read_bit(const uint8_t *data, int *bit_pos) {
    int byte_idx = *bit_pos >> 3;
    int bit_idx = *bit_pos & 7;
    (*bit_pos)++;
    return (data[byte_idx] >> bit_idx) & 1;
}

/* Decode one piece from Huffman stream. Returns piece type 1-7, sets *color (0=black,1=white), *promoted */
static int decode_piece(const uint8_t *data, int *bit_pos, int *color, int *promoted) {
    /* Huffman tree (bits read sequentially):
     * 0          -> PAWN (1)
     * 1,0,0      -> LANCE (2)
     * 1,0,1      -> KNIGHT (3)
     * 1,1,0      -> SILVER (4)
     * 1,1,1,0    -> GOLD (5)
     * 1,1,1,1,0  -> BISHOP (6)
     * 1,1,1,1,1  -> ROOK (7)
     */
    if (read_bit(data, bit_pos) == 0) {
        /* PAWN */
        *promoted = read_bit(data, bit_pos);
        *color = read_bit(data, bit_pos);
        return 1;
    }
    if (read_bit(data, bit_pos) == 0) {
        /* 10x -> LANCE or KNIGHT */
        if (read_bit(data, bit_pos) == 0) {
            /* 100 -> LANCE */
            *promoted = read_bit(data, bit_pos);
            *color = read_bit(data, bit_pos);
            return 2;
        }
        /* 101 -> KNIGHT */
        *promoted = read_bit(data, bit_pos);
        *color = read_bit(data, bit_pos);
        return 3;
    }
    if (read_bit(data, bit_pos) == 0) {
        /* 110 -> SILVER */
        *promoted = read_bit(data, bit_pos);
        *color = read_bit(data, bit_pos);
        return 4;
    }
    if (read_bit(data, bit_pos) == 0) {
        /* 1110 -> GOLD (no promoted bit) */
        *promoted = 0;
        *color = read_bit(data, bit_pos);
        return 5;
    }
    if (read_bit(data, bit_pos) == 0) {
        /* 11110 -> BISHOP */
        *promoted = read_bit(data, bit_pos);
        *color = read_bit(data, bit_pos);
        return 6;
    }
    /* 11111 -> ROOK */
    *promoted = read_bit(data, bit_pos);
    *color = read_bit(data, bit_pos);
    return 7;
}

/* Unpack a single 32-byte YaneuraOu packed SFEN into an SFEN string.
 * Returns length of string written, or -1 on error. */
static int unpack_psfen(const uint8_t *packed, char *out, int out_size) {
    int bit_pos = 0;
    char *p = out;
    char *end = out + out_size - 1;

    /* Turn */
    int turn_bit = read_bits(packed, &bit_pos, 1);
    char turn_char = turn_bit ? 'w' : 'b';

    /* King positions (7 bits each) */
    int king0_pos = read_bits(packed, &bit_pos, 7); /* black king (sente) */
    int king1_pos = read_bits(packed, &bit_pos, 7); /* white king (gote) */

    /* Decode 79 non-king squares */
    char board[81];
    memset(board, 0, sizeof(board));
    int board_piece_count = 2; /* 2 kings already counted */
    for (int i = 0; i < 81; i++) {
        if (i == king0_pos) {
            board[i] = 'K';
            continue;
        }
        if (i == king1_pos) {
            board[i] = 'k';
            continue;
        }
        int occupied = read_bit(packed, &bit_pos);
        if (occupied) {
            int color, promoted;
            int piece = decode_piece(packed, &bit_pos, &color, &promoted);
            if (piece < 1 || piece > 7) return -1;
            char c = color ? piece_lower[piece] : piece_upper[piece];
            if (promoted) {
                /* Store as two chars: '+' and piece */
                board[i] = '+';
                /* We need a way to store promoted. Use negative to flag. */
                board[i] = -(c); /* hack: negative means promoted */
            } else {
                board[i] = c;
            }
            board_piece_count++;
        }
    }

    /* Shogi always has exactly 40 pieces total */
    int hand_piece_count = 40 - board_piece_count;

    /* Build board string */
    for (int rank = 0; rank < 9; rank++) {
        if (rank > 0) {
            if (p >= end) return -1;
            *p++ = '/';
        }
        int empty = 0;
        for (int file = 0; file < 9; file++) {
            char c = board[rank * 9 + file];
            if (c == 0) {
                empty++;
            } else {
                if (empty > 0) {
                    if (p >= end) return -1;
                    *p++ = '0' + empty;
                    empty = 0;
                }
                if (c < 0) {
                    /* promoted piece */
                    if (p + 1 >= end) return -1;
                    *p++ = '+';
                    *p++ = (char)(-c);
                } else {
                    if (p >= end) return -1;
                    *p++ = c;
                }
            }
        }
        if (empty > 0) {
            if (p >= end) return -1;
            *p++ = '0' + empty;
        }
    }

    /* Turn */
    if (p + 2 >= end) return -1;
    *p++ = ' ';
    *p++ = turn_char;
    *p++ = ' ';

    /* Decode hand pieces - exactly hand_piece_count pieces */
    char hand_buf[64];
    int hand_len = 0;
    for (int h = 0; h < hand_piece_count && bit_pos + 3 <= 256; h++) {
        int color, promoted;
        int piece = decode_piece(packed, &bit_pos, &color, &promoted);
        if (piece < 1 || piece > 7 || bit_pos > 256) break;
        char c = color ? piece_lower[piece] : piece_upper[piece];
        if (hand_len < 63) hand_buf[hand_len++] = c;
    }

    /* Write hand string (compact duplicates) */
    if (hand_len == 0) {
        if (p >= end) return -1;
        *p++ = '-';
    } else {
        int i = 0;
        while (i < hand_len) {
            int count = 1;
            while (i + count < hand_len && hand_buf[i + count] == hand_buf[i])
                count++;
            if (count > 1) {
                if (p >= end) return -1;
                if (count >= 10) {
                    *p++ = '0' + count / 10;
                    if (p >= end) return -1;
                }
                *p++ = '0' + count % 10;
            }
            if (p >= end) return -1;
            *p++ = hand_buf[i];
            i += count;
        }
    }

    /* Ply */
    if (p + 2 >= end) return -1;
    *p++ = ' ';
    *p++ = '1';

    *p = '\0';
    return (int)(p - out);
}

/* Python wrapper: unpack_psfen_batch(numpy_array_of_uint8_32xN) -> list of sfen strings */
static PyObject* py_unpack_batch(PyObject *self, PyObject *args) {
    Py_buffer buf;
    if (!PyArg_ParseTuple(args, "y*", &buf))
        return NULL;

    int n = (int)(buf.len / 32);
    const uint8_t *data = (const uint8_t *)buf.buf;

    PyObject *result = PyList_New(n);
    if (!result) {
        PyBuffer_Release(&buf);
        return NULL;
    }

    char sfen[256];
    for (int i = 0; i < n; i++) {
        int len = unpack_psfen(data + i * 32, sfen, sizeof(sfen));
        if (len > 0) {
            PyList_SET_ITEM(result, i, PyUnicode_FromStringAndSize(sfen, len));
        } else {
            Py_INCREF(Py_None);
            PyList_SET_ITEM(result, i, Py_None);
        }
    }

    PyBuffer_Release(&buf);
    return result;
}

static PyMethodDef methods[] = {
    {"unpack_batch", py_unpack_batch, METH_VARARGS,
     "Unpack batch of 32-byte YaneuraOu packed SFENs to SFEN strings.\n"
     "Input: bytes object (N*32 bytes). Returns: list of N sfen strings."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "sfen_unpack_c", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_sfen_unpack_c(void) {
    return PyModule_Create(&module);
}

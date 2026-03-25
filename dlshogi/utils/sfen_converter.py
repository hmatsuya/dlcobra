#!/usr/bin/python3
'Converter between sfen string and YaneuraOu packed sfen for shogi'

# Source: https://github.com/ingktkhk/sfen_converter (Public Domain)

## Bitstreams
def int2bits(x, n):
    for i in range(n):
        yield (x >> i) & 1

def bits2int8(g):
    y = 0
    for i,x in enumerate(g):
        j = i & 7
        y |= x << j
        if j == 7:
            yield y
            y = 0
    if j != 7:
        yield y

def int8s2bits(l):
    for i in l:
        yield from int2bits(i, 8)

def bits2int(gen, n):
    x = 0
    for i in range(n):
        x |= next(gen) << i
    return x

## Huffman tables
huffman_invert = {
    0: 'Pp',
    4: 'Ll',
    5: 'Nn',
    6: 'Ss',
    14: 'Gg',
    30: 'Bb',
    31: 'Rr'
}

def bits2piece(gen):
    sym = None
    x = 0
    for i,b in enumerate(gen):
        x = x << 1 | b
        try:
            sym = huffman_invert[x]
            break
        except KeyError:
            pass
    if not sym:
        raise StopIteration
    promoted = False if sym == 'Gg' else next(gen)
    sym = sym[next(gen)]
    return "+" + sym if promoted else sym

def unpack_compact_row(row):
    zrl = 0
    for s in row:
        if not s:
            zrl += 1
        else:
            if zrl:
                yield str(zrl)
                zrl = 0
            yield s
    if zrl:
        yield str(zrl)

def unpack_board(flatboard):
    return "/".join("".join(s for s in unpack_compact_row(flatboard[y * 9 : y * 9 + 9])) for y in range(9))

def unpack_compact_hands(hands):
    if not hands:
        yield '-'
    else:
        n, q = 0, None
        for p in hands:
            if q is None:
                n, q = 1, p
            elif q == p:
                n += 1
            elif q:
                if 1 < n:
                    yield str(n)
                yield q
                n, q = 1, p
        if q:
            if 1 < n:
                yield str(n)
            yield q

def unpack_hands(hands):
    return "".join(unpack_compact_hands(hands))

def unpack(psfen, ply=1):
    gen = int8s2bits(psfen)
    turn = 'w' if next(gen) else 'b'
    king = [ (bits2int(gen, 7), 'K'), (bits2int(gen, 7), 'k') ]
    king.sort()
    board = list(bits2piece(gen) if next(gen) else 0 for i in range(79))
    for i, s in king:
        board.insert(i, s)
    hands = []
    while True:
        try:
            hands.append(bits2piece(gen))
        except StopIteration:
            break
    return " ".join((unpack_board(board), turn, unpack_hands(hands), str(ply)))

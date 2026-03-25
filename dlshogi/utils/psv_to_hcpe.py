"""Convert YaneuraOu PSV (.bin) files to HCPE format.

Uses C extension sfen_unpack_c for fast YaneuraOu packed SFEN decoding,
then cshogi to re-encode as HCP (Apery/dlshogi format).
Processes in chunks to handle large files without excessive memory use.
"""
from cshogi import *
import numpy as np
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from sfen_unpack_c import unpack_batch


def main():
    parser = argparse.ArgumentParser(description='Convert YaneuraOu PSV to HCPE')
    parser.add_argument('psv', help='Input PSV (.bin) file')
    parser.add_argument('hcpe', help='Output HCPE file')
    parser.add_argument('--chunk-size', type=int, default=100000,
                        help='Records per chunk (default: 100000)')
    args = parser.parse_args()

    total = os.path.getsize(args.psv) // np.dtype(PackedSfenValue).itemsize
    if total == 0:
        print('input num = 0, skipping')
        sys.exit(1)

    print(f'input num = {total}')
    print(f'chunk_size = {args.chunk_size}')

    psvs = np.memmap(args.psv, dtype=PackedSfenValue, mode='r')
    board = Board()
    total_ok = 0
    total_err = 0
    t0 = time.time()

    with open(args.hcpe, 'wb') as f_out:
        for start in range(0, total, args.chunk_size):
            end = min(start + args.chunk_size, total)
            chunk = np.array(psvs[start:end])  # copy to memory

            # Batch unpack YaneuraOu packed SFENs to SFEN strings
            sfen_list = unpack_batch(chunk['sfen'].tobytes())

            hcpes = np.zeros(len(chunk), dtype=HuffmanCodedPosAndEval)
            num_ok = 0

            for j in range(len(chunk)):
                sfen = sfen_list[j]
                if sfen is None:
                    total_err += 1
                    continue
                try:
                    board.set_sfen(sfen)
                    hcpe = hcpes[num_ok]
                    board.to_hcp(hcpe['hcp'])
                    hcpe['eval'] = chunk[j]['score']
                    hcpe['bestMove16'] = move16_from_psv(chunk[j]['move'])
                    gr = chunk[j]['game_result']
                    if gr == 1:
                        hcpe['gameResult'] = board.turn + 1
                    elif gr == -1:
                        hcpe['gameResult'] = 2 - board.turn
                    num_ok += 1
                except Exception:
                    total_err += 1

            hcpes[:num_ok].tofile(f_out)
            total_ok += num_ok

            elapsed = time.time() - t0
            done = end
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f'  [{done}/{total}] {rate:.0f} rec/s, '
                  f'ok={total_ok}, err={total_err}, '
                  f'ETA={eta/60:.1f}min', flush=True)

    elapsed = time.time() - t0
    print(f'position num = {total_ok}')
    print(f'error num = {total_err}')
    if total > 0:
        print(f'error rate = {total_err / total:.6f}')
    print(f'time = {elapsed:.1f}s ({total / elapsed:.0f} rec/s)')


if __name__ == '__main__':
    main()

import cshogi
import numpy as np
import os
import glob
import lzma
import argparse

def flip_sfen(sfen):
    board, others = sfen.split(' ', 1)
    lines = board.split('/')
    flipped_lines = []
    for line in lines:
        flipped_line = ''
        promoted = False
        for char in line:
            if char == '+':
                promoted = True
                continue
            if promoted:
                flipped_line = '+' + char + flipped_line
            else:
                flipped_line = char + flipped_line
            promoted = False
        flipped_lines.append(flipped_line)

    flipped_sfen = '/'.join(flipped_lines) + ' ' + others
    return flipped_sfen

def flip_sfen_square(sq):
    if len(sq) != 2:
        raise ValueError(f'Invalid SFEN square: {sq}')
    if int(sq[0]) < 1 or int(sq[0]) > 9:
        raise ValueError(f'Invalid SFEN square: {sq}')
    if sq[1] not in "abcdefghi":
        raise ValueError(f'Invalid SFEN square: {sq}')

    flipped = str(10 - int(sq[0]))
    flipped += chr(ord('i') - ord(sq[1]) + ord('a'))

    return flipped

def flip_sfen_move(move):
    move = move.strip()
    flipped = ''
    if move[1] == '*':
        flipped = move[0:2]
    else:
        flipped = flip_sfen_square(move[0:2])

    flipped += flip_sfen_square(move[2:4])

    if move[-1] == '+':
        flipped += '+'

    return flipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csa_dir')
    parser.add_argument('out_dir')
    parser.add_argument('flip_ply', default=60, type=int, nargs='?')
    args = parser.parse_args()

    csa_file_list = glob.glob(os.path.join(args.csa_dir, '**', '*.txt*'), recursive=True)
    os.makedirs(args.out_dir, exist_ok=True)

    hcpes = np.zeros(152770943*2, cshogi.HuffmanCodedPosAndEval)

    board = cshogi.Board()
    kif_num = 0
    position_num = 0
    for filepath in csa_file_list:
        print(filepath)
        p = 0
        if filepath.endswith('.xz'):
            input_file = lzma.open(filepath, 'rt')
            filepath = filepath[:-3]
        elif filepath.endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                # Assuming the zip contains a single file, extract it
                extracted_files = zip_ref.namelist()
                if len(extracted_files) != 1:
                    raise ValueError("Zip file must contain exactly one file.")
                extracted_file = extracted_files[0]
                zip_ref.extract(extracted_file, os.path.dirname(filepath))
                filepath = os.path.join(os.path.dirname(filepath), extracted_file)
            input_file = open(filepath, 'r')
        elif filepath.endswith('.gz'):
            import gzip
            input_file = gzip.open(filepath, 'rt')
            filepath = filepath[:-3]
        else:
            input_file = open(filepath, 'r')

        ply = -1
        sfen = ''
        sfen_move = ''
        for line in input_file:

            # exemple:
            # sfen l3k2nl/1r1sg1g2/2n1psbp1/2pp1pp1p/pp2P2P1/2PP2P1P/PP1S1P3/2GBGS1R1/LN1K3NL b - 0
            # move 2i3g
            # score -20
            # ply 33
            # result -1
            # e

            hcpe = hcpes[p]
            if line.strip() == 'e':
                p += 1
                position_num += 1

                if ply >= args.flip_ply:
                    # write out flipped position
                    hcpes[p] = hcpes[p-1] # copy values

                    board.set_sfen(flip_sfen(sfen))
                    board.to_hcp(hcpe['hcp'])

                    flipped_sfen_move = flip_sfen_move(sfen_move.strip())
                    move = board.move_from_usi(flipped_sfen_move)
                    hcpe['bestMove16'] = cshogi.move16(move)

                    p += 1

                # initialize
                ply = -1
            else:
                (label, data) = line.split(' ', 1)
                data = data.strip()
            if label == 'sfen':
                sfen = data
                board.set_sfen(data)
                assert(data.strip() == board.sfen().strip())
                board.to_hcp(hcpe['hcp'])
            elif label == 'move':
                move = board.move_from_usi(data.strip())
                assert(data == cshogi.move_to_usi(move))
                hcpe['bestMove16'] = cshogi.move16(move)
                sfen_move = data
            elif label == 'score':
                hcpe['eval'] = int(data)
            elif label == 'ply':
                ply = int(data)
            elif label == 'result':
                if int(data) == 0:
                    hcpe['gameResult'] = 0
                else:
                    winner = board.turn
                    if int(data) == -1:
                        winner = winner ^ 1
                    hcpe['gameResult'] = winner + 1

        hcpes[:p].tofile(os.path.join(args.out_dir, os.path.splitext(os.path.basename(filepath))[0] + '.hcpe'))
        position_num += p

    print('kif_num', kif_num)
    print('position_num', position_num)

if __name__ == "__main__":
    main()
from cshogi import *
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('psv')
parser.add_argument('hcpe')
args = parser.parse_args()

psvs = np.fromfile(args.psv, dtype=PackedSfenValue)
hcpes = np.zeros(len(psvs), dtype=HuffmanCodedPosAndEval)

print(f'input num = {len(psvs)}')

board = Board()
num_positions = 0
for index, (psv, hcpe) in enumerate(zip(psvs, hcpes)):
    try:
        if not board.set_psfen(psv['sfen']):
            raise ValueError('failed to set sfen')
        if not board.is_ok():
            raise ValueError('board is not ok')
        hcpe['eval'] = psv['score']
        board.to_hcp(hcpe['hcp'])

        if not isinstance(hcpe['eval'], int):
            raise ValueError('eval is not int')
        
        hcpe['bestMove16'] = move16_from_psv(psv['move'])
        if not isinstance(hcpe['bestMove16'], int):
            raise ValueError('bestMove16 is not int')
        move = board.move_from_move16(hcpe['bestMove16'])
        if not board.is_legal(move):
            raise ValueError('illegal move')

        game_result = psv['game_result']
        # gameResult -> 0: DRAW, 1: BLACK_WIN, 2: WHITE_WIN
        if game_result == 1:
            hcpe['gameResult'] = board.turn + 1
        elif game_result == -1:
            hcpe['gameResult'] = 2 - board.turn
        if hcpe['gameResult'] not in [0, 1, 2]:
            raise ValueError('gameResult is not 0, 1, 2')

        num_positions += 1
    except(Exception) as e:
        print(index, e, psv)
        # np.delete(hcpes, num_positions, axis=0)

print(f'position num = {num_positions}')
print(f'output num = {len(hcpes)}')
print(f'error rate = {1 - num_positions / len(psvs)}')

hcpes.tofile(args.hcpe)

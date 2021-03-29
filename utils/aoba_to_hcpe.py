from cshogi import *
from cshogi import CSA
import numpy as np
import os
import glob
import lzma
import math
import argparse
import re
import pandas as pd
from scipy.special import logit

parser = argparse.ArgumentParser()
parser.add_argument('csa_dir')
parser.add_argument('out_dir')
args = parser.parse_args()

def get_values(comments):
    p = re.compile('v\=([\d\.]+)')
    values = []
    for c in comments:
        m = p.search(c.decode("utf-8"))
        if m is not None:
            values.append(float(m[1]))

    return values

# Moving averate
def value_ewma(values, n=10):
    # all values to black value
    black = [v if (p % 2) == 0 else 1-v for p, v in enumerate(values)]
    black = black + [black[-1]] * (n-1)

    # ewma
    ewma = pd.Series(black, dtype=np.float64).rolling(window=n, min_periods=1).mean().tolist()[n-1:]
    assert(len(ewma) == len(values))

    # even-th value to white value
    return [min(1.0, v) if (p%2) == 0 else max(0, 1-v) for p, v in enumerate(ewma)]

# inverse of value_to_score() https://github.com/TadaoYamaoka/DeepLearningShogi/blob/2134eedf3e1d8f37bab12444e55763a83eff1027/cppshogi/cppshogi.h
def value_to_score(values, alpha=1/0.0013226):
    score = logit(values) * alpha
    for i, s in enumerate(score):
        if s in [float('inf'), -float('inf')]:
            score[i] = 30000 * np.sign(s)
    return score

csa_file_list = glob.glob(os.path.join(args.csa_dir, '**', '*.csa*'), recursive=True)
os.makedirs(args.out_dir, exist_ok=True)

hcpes = np.zeros(10000*512, HuffmanCodedPosAndEval)

board = Board()
kif_num = 0
position_num = 0
for filepath in csa_file_list:
    print(filepath)
    p = 0
    if filepath.endswith('.xz'):
        file = lzma.open(filepath, 'rt')
        filepath = filepath[:-3]
    else:
        file = filepath
    for kif in CSA.Parser.parse_file(file):
        if kif.endgame not in ('%TORYO', '%SENNICHITE', '%KACHI', '%HIKIWAKE', '%CHUDAN') or len(kif.moves) <= 30:
            continue

        # parse values
        values = get_values(kif.comments)
        if len(values) != len(kif.moves):
            print(len(values), len(kif.moves), flush=True)
            continue
        ewma = value_ewma(values)
        score = value_to_score(ewma)

        kif_num += 1
        board.set_sfen(kif.sfen)
        # 30手までで最善手以外が指された手番を見つける
        start = -1
        for i, (move, comment) in enumerate(zip(kif.moves, kif.comments)):
            comments = comment.decode('ascii').split(',')
            if comments[0].startswith('v='):
                candidates = comments[1:]
            else:
                candidates = comments
            if board.move_from_csa(candidates[1]) != move:
                start = i
            if i >= 29:
                break
            board.push(move)

        # 最後に最善手以外が指された局面から開始する
        board.set_sfen(kif.sfen)
        start_p = p
        for i, move in enumerate(kif.moves):
            if i <= start:
                board.push(move)
                continue
            # 5手詰みチェック
            if board.mate_move(5) != 0:
                if kif.win != board.turn + 1:
                    # 詰みを見逃して逆転したゲームの結果を修正
                    hcpes[start_p:p]['gameResult'] = board.turn + 1
                break
            hcpe = hcpes[p]
            board.to_hcp(hcpe['hcp'])
            hcpe['bestMove16'] = move16(move)
            hcpe['gameResult'] = kif.win
            hcpe['eval'] = round(score[i])
            p += 1
            board.push(move)

    hcpes[:p].tofile(os.path.join(args.out_dir, os.path.splitext(os.path.basename(filepath))[0] + '.hcpe'))
    position_num += p

print('kif_num', kif_num)
print('position_num', position_num)

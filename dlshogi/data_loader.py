import numpy as np
import torch
import torchvision

from dlshogi.common import *
from dlshogi import cppshogi

import os
from concurrent.futures import ThreadPoolExecutor

import logging

def hflip_sq(sq):
    row = sq % 9
    assert(row < 9)
    assert(row >= 0)

    col = sq // 9
    assert(col < 9)
    assert(col >= 0)

    col = 8 - col

    sq = col * 9 + row
    assert(sq >= 0)
    assert(sq < 81)

    return  int(sq)

def hflip_move16(move):
    # see: move.hpp : 30
    # xxxxxxxx x1111111  移動先
    # xx111111 1xxxxxxx  移動元。駒打ちの際には、PieceType + SquareNum - 1
    # x1xxxxxx xxxxxxxx  1 なら成り
    to_sq = move & 0b1111111
    from_sq = (move >> 7) & 0b1111111

    # from sq
    if from_sq < 81:
        from_sq = hflip_sq(from_sq)
        move = move & 0b1100000001111111
        move = move | (from_sq << 7)

    # to sq
    to_sq = hflip_sq(to_sq)
    move = move & 0b1111111110000000
    move = move | to_sq

    return move

class DataLoader:

    # move direction
    hflip_direction = [
        UP, UP_RIGHT, UP_LEFT, RIGHT, LEFT, DOWN, DOWN_RIGHT, DOWN_LEFT, UP2_RIGHT, UP2_LEFT,
        UP_PROMOTE, UP_RIGHT_PROMOTE, UP_LEFT_PROMOTE, RIGHT_PROMOTE, LEFT_PROMOTE, DOWN_PROMOTE, DOWN_RIGHT_PROMOTE, DOWN_LEFT_PROMOTE, UP2_RIGHT_PROMOTE, UP2_LEFT_PROMOTE
    ]
    hflip_move_labels = np.array(hflip_direction + list(range(20, 27)))
    hflip_squares = np.arange(81).reshape([9, 9])[::-1,:].flatten()
    hflip_idx  = np.repeat(hflip_move_labels, len(hflip_squares)) * 81 + np.tile(hflip_squares, len(hflip_move_labels))

    @staticmethod
    def load_files(files):
        data = []
        for path in files:
            if os.path.exists(path):
                logging.info(path)
                data.append(np.fromfile(path, dtype=HuffmanCodedPosAndEval))
            else:
                logging.warn('{} not found, skipping'.format(path))
        return np.concatenate(data)

    def __init__(self, data, batch_size, device, shuffle=False, hflip=0.0):
        self.data = data
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

        self.torch_features1 = torch.empty((batch_size, FEATURES1_NUM, 9, 9), dtype=torch.float32, pin_memory=True)
        self.torch_features2 = torch.empty((batch_size, FEATURES2_NUM, 9, 9), dtype=torch.float32, pin_memory=True)
        self.torch_move = torch.empty((batch_size), dtype=torch.int64, pin_memory=True)
        self.torch_result = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)
        self.torch_value = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)

        self.features1 = self.torch_features1.numpy()
        self.features2 = self.torch_features2.numpy()
        self.move = self.torch_move.numpy()
        self.result = self.torch_result.numpy().reshape(-1)
        self.value = self.torch_value.numpy().reshape(-1)

        self.i = 0
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.hflip=hflip

    def mini_batch(self, hcpevec):
        cppshogi.hcpe_decode_with_value(hcpevec, self.features1, self.features2, self.move, self.result, self.value)

        if self.hflip > 0:
            hflips = np.random.binomial(1, self.hflip, self.batch_size)

            for i, flip in enumerate(hflips):
                if flip:
                    self.features1[i] == self.hflip_idx[self.features1[i]]
                    hcpevec[i]['bestMove16'] = hflip_move16(hcpevec[i]['bestMove16'])

        if self.device.type == 'cpu':
            return (self.torch_features1.clone(),
                    self.torch_features2.clone(),
                    self.torch_move.clone(),
                    self.torch_result.clone(),
                    self.torch_value.clone()
                    )
        else:
            return (self.torch_features1.to(self.device),
                    self.torch_features2.to(self.device),
                    self.torch_move.to(self.device),
                    self.torch_result.to(self.device),
                    self.torch_value.to(self.device)
                    )

    def sample(self):
        return self.mini_batch(np.random.choice(self.data, self.batch_size, replace=False))

    def sample_test(self):
        hcpevec = np.random.choice(self.data, self.batch_size, replace=False)
        return self.mini_batch(hcpevec) + (hcpevec,)

    def pre_fetch(self):
        hcpevec = self.data[self.i:self.i+self.batch_size]
        self.i += self.batch_size
        if len(hcpevec) < self.batch_size:
            return

        self.f = self.executor.submit(self.mini_batch, hcpevec)

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            np.random.shuffle(self.data)
        self.pre_fetch()
        return self

    def __next__(self):
        if self.i > len(self.data):
            raise StopIteration()

        result = self.f.result()
        self.pre_fetch()

        return result

class Hcpe2DataLoader(DataLoader):
    @staticmethod
    def load_files(files):
        data = []
        for path in files:
            if os.path.exists(path):
                logging.info(path)
                data.append(np.fromfile(path, dtype=HuffmanCodedPosAndEval2))
            else:
                logging.warn('{} not found, skipping'.format(path))
        return np.concatenate(data)

    def __init__(self, data, batch_size, device, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

        self.torch_features1 = torch.empty((batch_size, FEATURES1_NUM, 9, 9), dtype=torch.float32, pin_memory=True)
        self.torch_features2 = torch.empty((batch_size, FEATURES2_NUM, 9, 9), dtype=torch.float32, pin_memory=True)
        self.torch_move = torch.empty((batch_size), dtype=torch.int64, pin_memory=True)
        self.torch_result = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)
        self.torch_value = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)
        self.torch_aux = torch.empty((batch_size, 2), dtype=torch.float32, pin_memory=True)

        self.features1 = self.torch_features1.numpy()
        self.features2 = self.torch_features2.numpy()
        self.move = self.torch_move.numpy()
        self.result = self.torch_result.numpy().reshape(-1)
        self.value = self.torch_value.numpy().reshape(-1)
        self.aux = self.torch_aux.numpy()

        self.i = 0
        self.executor = ThreadPoolExecutor(max_workers=1)

    def mini_batch(self, hcpevec):
        cppshogi.hcpe2_decode_with_value(hcpevec, self.features1, self.features2, self.move, self.result, self.value, self.aux)

        return (self.torch_features1.to(self.device),
                self.torch_features2.to(self.device),
                self.torch_move.to(self.device),
                self.torch_result.to(self.device),
                self.torch_value.to(self.device),
                self.torch_aux.to(self.device)
                )

# 評価値から勝率への変換
def score_to_value(score, a):
    return 1.0 / (1.0 + np.exp(-score / a))

class Hcpe3DataLoader(DataLoader):
    @staticmethod
    def load_files(files, use_average=False, use_evalfix=False, temperature=1.0):
        if use_evalfix:
            from scipy.optimize import curve_fit

        actual_len = 0
        for path in files:
            if os.path.exists(path):
                if use_evalfix:
                    eval, result = cppshogi.hcpe3_prepare_evalfix(path)
                    if (eval == 0).all():
                        a = 0
                        logging.info('{}, skip evalfix'.format(path))
                    else:
                        popt, _ = curve_fit(score_to_value, eval, result, p0=[300.0])
                        a = popt[0]
                        logging.info('{}, a={}'.format(path, a))
                else:
                    a = 0
                    logging.info(path)
                sum_len, len_ = cppshogi.load_hcpe3(path, use_average, a, temperature)
                if len_ == 0:
                    raise RuntimeError('read error {}'.format(path))
                actual_len += len_
            else:
                logging.warn('{} not found, skipping'.format(path))
        return sum_len, actual_len

    def __init__(self, data, batch_size, device, shuffle=False, hflip=0.0):
        self.data = data
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

        self.torch_features1 = torch.empty((batch_size, FEATURES1_NUM, 9, 9), dtype=torch.float32, pin_memory=True)
        self.torch_features2 = torch.empty((batch_size, FEATURES2_NUM, 9, 9), dtype=torch.float32, pin_memory=True)
        self.torch_probability = torch.empty((batch_size, 9*9*MAX_MOVE_LABEL_NUM), dtype=torch.float32, pin_memory=True)
        self.torch_result = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)
        self.torch_value = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)

        self.features1 = self.torch_features1.numpy()
        self.features2 = self.torch_features2.numpy()
        self.probability = self.torch_probability.numpy()
        self.result = self.torch_result.numpy().reshape(-1)
        self.value = self.torch_value.numpy().reshape(-1)

        self.i = 0
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.hflip = hflip

    def mini_batch(self, index):
        cppshogi.hcpe3_decode_with_value(index, self.features1, self.features2, self.probability, self.result, self.value)

        if self.hflip > 0:
            hflips = np.random.binomial(1, self.hflip, self.batch_size)
            for i, flip in enumerate(hflips):
                if flip:
                    self.features1[i] = np.fliplr(self.features1[i])
                    self.probability[i] = self.probability[i][self.hflip_idx]


        return (self.torch_features1.to(self.device),
                self.torch_features2.to(self.device),
                self.torch_probability.to(self.device),
                self.torch_result.to(self.device),
                self.torch_value.to(self.device)
                )

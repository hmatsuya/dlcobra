"""DataModule that streams HCPE files one at a time to avoid OOM.

Loads a single HCPE file into RAM, trains on it for one epoch,
then swaps to the next file. Cycles through all files over max_epochs.
"""
import glob
import logging

import lightning.pytorch as pl
import numpy as np
import torch
from cshogi import HuffmanCodedPosAndEval
from torch.utils.data import DataLoader, Dataset

from dlshogi import cppshogi
from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM
from dlshogi.ptl import collate

logger = logging.getLogger(__name__)


class SingleFileHcpeDataset(Dataset):
    """Dataset backed by a single HCPE file loaded into memory."""

    def __init__(self, path):
        logger.info(f"Loading {path}")
        self.hcpe = np.fromfile(path, dtype=HuffmanCodedPosAndEval)
        logger.info(f"  positions: {len(self.hcpe)}")

    def __len__(self):
        return len(self.hcpe)

    def __getitems__(self, indexes):
        batch_size = len(indexes)
        hcpevec = self.hcpe[indexes]

        features1 = torch.empty(
            (batch_size, FEATURES1_NUM, 9, 9), dtype=torch.float32, pin_memory=True
        )
        features2 = torch.empty(
            (batch_size, FEATURES2_NUM, 9, 9), dtype=torch.float32, pin_memory=True
        )
        move = torch.empty((batch_size,), dtype=torch.int64, pin_memory=True)
        result = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)
        value = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)

        cppshogi.hcpe_decode_with_value(
            hcpevec,
            features1.numpy(),
            features2.numpy(),
            move.numpy(),
            result.numpy(),
            value.numpy(),
        )
        return features1, features2, move, result, value


class HcpeDataModule(pl.LightningDataModule):
    """Loads one HCPE file per epoch, cycling through all files."""

    def __init__(
        self,
        train_files,
        val_files,
        batch_size=1024,
        val_batch_size=1024,
        # kept for config compatibility
        use_average=False,
        use_evalfix=False,
        temperature=1.0,
        patch=None,
        cache=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Expand globs once
        self._train_paths = []
        for pattern in train_files:
            self._train_paths.extend(sorted(glob.glob(pattern)))
        self._file_index = 0
        self._train_dataset = None

    def setup(self, stage: str):
        if stage in ("fit", "validate"):
            from dlshogi.ptl import HcpeDataset
            self.val_dataset = HcpeDataset(self.hparams.val_files)
        if stage in ("test", "predict"):
            from dlshogi.ptl import HcpeDataset
            self.val_dataset = HcpeDataset(self.hparams.val_files)

    def _load_next_train_file(self):
        """Load the next HCPE file (cycling)."""
        idx = self._file_index % len(self._train_paths)
        path = self._train_paths[idx]
        # Free previous dataset
        self._train_dataset = None
        self._train_dataset = SingleFileHcpeDataset(path)
        logger.info(f"Train file [{idx+1}/{len(self._train_paths)}]: {path}")
        self._file_index += 1

    def train_dataloader(self):
        """Called every epoch (reload_dataloaders_every_n_epochs=1)."""
        self._load_next_train_file()
        return DataLoader(
            self._train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            collate_fn=collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            collate_fn=collate,
        )

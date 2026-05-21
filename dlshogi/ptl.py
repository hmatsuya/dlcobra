import logging
import os
from collections import defaultdict

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm, TQDMProgressBar
from lightning.pytorch.cli import LightningCLI
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn, SWALR
from torch.utils.data import DataLoader, Dataset

from dlshogi import cppshogi, serializers
from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM, MAX_MOVE_LABEL_NUM
from dlshogi.data_loader import DataLoader as HcpeDataLoader
from dlshogi.data_loader import Hcpe3DataLoader, Hdf5DataLoader
from dlshogi.network.policy_value_network import policy_value_network

logger = logging.getLogger(__name__)


class HcpeDataset(Dataset):
    def __init__(self, files):
        logger = logging.getLogger("lightning.pytorch.core")
        logger.info("Loading HcpeDataset")
        self.hcpe = HcpeDataLoader.load_files(files, logger)
        logger.info("position num = {}".format(len(self.hcpe)))

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
        move = torch.empty((batch_size), dtype=torch.int64, pin_memory=True)
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


class Hdf5HcpeDataset(HcpeDataset):
    def __init__(self, files):
        logger = logging.getLogger("lightning.pytorch.core")
        logger.info("Loading HDF5 HcpeDataset")
        self.hcpe = Hdf5DataLoader.load_files(files, logger)
        logger.info("position num = {}".format(len(self.hcpe)))

    def __len__(self):
        return len(self.hcpe)

    def __getitems__(self, indexes):
        batch_size = len(indexes)
        hcpevec = self.hcpe[indexes].compute()

        features1 = torch.empty(
            (batch_size, FEATURES1_NUM, 9, 9), dtype=torch.float32, pin_memory=True
        )
        features2 = torch.empty(
            (batch_size, FEATURES2_NUM, 9, 9), dtype=torch.float32, pin_memory=True
        )
        move = torch.empty((batch_size), dtype=torch.int64, pin_memory=True)
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

    # when destroyed
    def __del__(self):
        Hdf5DataLoader.close_files()


class Hcpe3Dataset(Dataset):
    def __init__(self, files, use_average, use_evalfix, temperature, patch, cache):
        self.files = files
        self.use_average = use_average
        self.use_evalfix = use_evalfix
        self.temperature = temperature
        self.patch = patch
        self.cache = cache
        self.load()

    def load(self):
        logger = logging.getLogger("lightning.pytorch.core")
        logger.info("Loading Hcpe3Dataset")
        self.len, actual_len = Hcpe3DataLoader.load_files(
            self.files,
            self.use_average,
            self.use_evalfix,
            self.temperature,
            self.patch,
            self.cache,
            logger,
        )
        if self.use_average:
            logger.info("position num before preprocessing = {}".format(actual_len))
        logger.info("position num = {}".format(self.len))

    def __len__(self):
        return self.len

    def __getitems__(self, indexes):
        batch_size = len(indexes)
        indexes = np.array(indexes, dtype=np.uint64)

        features1 = torch.empty(
            (batch_size, FEATURES1_NUM, 9, 9), dtype=torch.float32, pin_memory=True
        )
        features2 = torch.empty(
            (batch_size, FEATURES2_NUM, 9, 9), dtype=torch.float32, pin_memory=True
        )
        probability = torch.empty(
            (batch_size, 9 * 9 * MAX_MOVE_LABEL_NUM),
            dtype=torch.float32,
            pin_memory=True,
        )
        result = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)
        value = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)

        cppshogi.hcpe3_decode_with_value(
            indexes,
            features1.numpy(),
            features2.numpy(),
            probability.numpy(),
            result.numpy(),
            value.numpy(),
        )

        return features1, features2, probability, result, value


def collate(data):
    return data


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_files,
        val_files,
        batch_size=1024,
        val_batch_size=1024,
        use_average=False,
        use_evalfix=False,
        temperature=1.0,
        patch=None,
        cache=None,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # self.train_dataset = Hdf5HcpeDataset(
            #     self.hparams.train_files,
            # )
            self.train_dataset = Hdf5DataLoader.load_files(self.hparams.train_files)
            self.val_dataset = HcpeDataset(self.hparams.val_files)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage == "predict":
            self.val_dataset = HcpeDataset(self.hparams.val_files)

    def train_dataloader(self):
        return Hdf5DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            device=torch.device('cuda'),
            shuffle=False,
            # collate_fn=collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.hparams.val_batch_size, collate_fn=collate
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.hparams.val_batch_size, collate_fn=collate
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.hparams.val_batch_size, collate_fn=collate
        )


def cross_entropy_loss_with_soft_target(pred, soft_targets):
    return torch.sum(-soft_targets * F.log_softmax(pred, dim=1), 1)


cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()


def accuracy(y, t):
    return (torch.max(y, 1)[1] == t).sum() / len(t)


def binary_accuracy(y, t):
    pred = y >= 0
    truth = t >= 0.5
    return pred.eq(truth).sum() / len(t)


class Model(pl.LightningModule):
    def __init__(
        self,
        network="resnet10_relu",
        val_lambda=0.333,
        val_lambda_decay_epoch=None,
        use_ema=False,
        update_bn=True,
        ema_start_epoch=1,
        ema_freq=250,
        ema_decay=0.9,
        lr_scheduler_interval="epoch",
        model_filename=None,
        resume_model=None,
        use_swa=False,
        swa_start_epoch=10,
        swa_lr=1e-4,
        compile_model=False,
        flip_augmentation=False,
        flip_ratio=0.5,
        kd_ratio=0.0,
        teacher_ckpt=None,
        teacher_network=None,
        teacher_temperature=2.0,
        # Symmetry Consistency Loss: forces model predictions on a board and its
        # horizontal mirror to be consistent (value identical, policy mirrored).
        # This acts as a regularizer that effectively doubles training signal
        # without requiring extra labelled data.
        sym_consistency_ratio=0.0,
        sym_consistency_warmup_steps=0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = policy_value_network(network)
        if resume_model:
            checkpoint = torch.load(resume_model, map_location="cpu")
            if "model" in checkpoint:
                # serializers.save_npz format
                self.model.load_state_dict(checkpoint["model"])
            else:
                # Lightning checkpoint format (state_dict with "model." or "model._orig_mod." prefix)
                state_dict = checkpoint.get("state_dict", checkpoint)
                stripped = {}
                for k, v in state_dict.items():
                    if k.startswith("model._orig_mod."):
                        stripped[k[len("model._orig_mod."):]] = v
                    elif k.startswith("model."):
                        stripped[k[len("model."):]] = v
                self.model.load_state_dict(stripped)
        if compile_model:
            self.model = torch.compile(self.model)
        # Teacher model for knowledge distillation
        self._teacher_model = None
        if kd_ratio > 0.0 and teacher_ckpt:
            teacher_net = policy_value_network(teacher_network if teacher_network else network)
            ckpt = torch.load(teacher_ckpt, map_location="cpu")
            # Lightning checkpoint: state_dict is under "state_dict" key, with "model." prefix
            state_dict = ckpt.get("state_dict", ckpt)
            # Strip "model." prefix if present (Lightning wraps model in self.model)
            # Also handle torch.compile case: "model._orig_mod." prefix
            stripped = {}
            for k, v in state_dict.items():
                if k.startswith("model._orig_mod."):
                    stripped[k[len("model._orig_mod."):]] = v
                elif k.startswith("model."):
                    stripped[k[len("model."):]] = v
            if stripped:
                teacher_net.load_state_dict(stripped)
            else:
                teacher_net.load_state_dict(state_dict)
            teacher_net.eval()
            teacher_net.requires_grad_(False)
            teacher_net = teacher_net.to(torch.bfloat16)
            self._teacher_model = teacher_net
        if use_ema:
            self.ema_model = AveragedModel(
                self.model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay)
            )
            self.ema_model.requires_grad_(False)
        self.validation_step_outputs = defaultdict(list)
        self.val_lambda = val_lambda
        self.use_swa = use_swa
        self.swa_start_epoch = swa_start_epoch
        self.swa_lr = swa_lr
        self.swa_model = None
        self.swa_scheduler = None

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        if self.use_swa:
            # Wrap the optimizer with SWA
            self.swa_scheduler = SWALR(optimizer, swa_lr=self.swa_lr)
        return optimizer

    def on_train_epoch_start(self):
        # update val_lambda
        if self.hparams.val_lambda_decay_epoch:
            self.val_lambda = max(
                0,
                self.hparams.val_lambda * (1 - self.current_epoch / self.hparams.val_lambda_decay_epoch)
            )
            self.log("val_lambda", self.val_lambda)
        if self.use_swa and self.current_epoch == self.swa_start_epoch:
            # Initialize SWA model
            self.swa_model = AveragedModel(self.model)

    def on_train_start(self):
        if self._teacher_model is not None:
            self._teacher_model = self._teacher_model.to(self.device)

    def training_step(self, batch, batch_idx):
        features1, features2, move, result, value = batch
        if self.hparams.flip_augmentation:
            from dlshogi.augmentation import apply_horizontal_flip
            features1, features2, move, result, value = apply_horizontal_flip(
                features1, features2, move, result, value,
                flip_ratio=self.hparams.flip_ratio,
            )
        y1, y2 = self.model(features1, features2)

        # Policy loss: mix CE and KD
        kd_ratio = self.hparams.kd_ratio
        loss1_ce = cross_entropy_loss(y1, move).mean()
        if kd_ratio > 0.0 and self._teacher_model is not None:
            T = self.hparams.teacher_temperature
            with torch.no_grad():
                t1, _ = self._teacher_model(features1.to(torch.bfloat16), features2.to(torch.bfloat16))
                t1 = t1.float()
            soft_targets = F.softmax(t1 / T, dim=1)
            loss1_kd = cross_entropy_loss_with_soft_target(y1 / T, soft_targets).mean() * (T * T)
            loss1 = (1.0 - kd_ratio) * loss1_ce + kd_ratio * loss1_kd
            self.log("train/policy_loss_kd", loss1_kd)
            self.log("train/policy_loss_mixed", loss1)
        else:
            loss1 = loss1_ce

        loss2 = bce_with_logits_loss(y2, result)
        loss3 = bce_with_logits_loss(y2, value)
        loss = (
            loss1
            + (1 - self.hparams.val_lambda) * loss2
            + self.hparams.val_lambda * loss3
        )

        # Symmetry Consistency Loss
        # Forces the model to produce consistent predictions for a board and its
        # horizontal mirror: value should be identical, policy should be mirrored.
        # Uses the correct move-label flip table from augmentation.py (not a naive
        # spatial flip), which properly handles direction encoding in the 2187-dim
        # policy vector (27 directions × 81 squares).
        sym_ratio = self.hparams.sym_consistency_ratio
        if sym_ratio > 0.0:
            # Linear warmup: ramp sym_ratio from 0 to target over warmup steps
            warmup = self.hparams.sym_consistency_warmup_steps
            if warmup > 0 and self.global_step < warmup:
                sym_ratio = sym_ratio * self.global_step / warmup

            from dlshogi.augmentation import flip_features1, flip_features2, flip_probability
            x1_flip = flip_features1(features1)
            x2_flip = flip_features2(features2)
            y1_flip, y2_flip = self.model(x1_flip, x2_flip)

            # Value consistency: v(board) == v(flipped_board)
            val_cons_loss = F.mse_loss(y2.sigmoid(), y2_flip.sigmoid())

            # Policy consistency: softmax(p(board)) == flip(softmax(p(flipped_board)))
            # flip_probability reorders the 2187-dim vector using the correct
            # direction+square mapping, not a naive spatial reshape.
            p_orig = F.softmax(y1, dim=1)
            p_flip_mirrored = flip_probability(F.softmax(y1_flip, dim=1))
            pol_cons_loss = F.mse_loss(p_orig, p_flip_mirrored)

            sym_loss = sym_ratio * (val_cons_loss + pol_cons_loss)
            loss = loss + sym_loss
            self.log("train/sym_val_cons_loss", val_cons_loss)
            self.log("train/sym_pol_cons_loss", pol_cons_loss)
            self.log("train/sym_loss", sym_loss)
            self.log("train/sym_ratio", sym_ratio)

        self.log("train/loss", loss)
        self.log("train/policy_loss", loss1_ce)
        self.log("train/result_loss", loss2)
        self.log("train/value_loss", loss3)
        self.log("train/policy_accuracy", accuracy(y1, move))
        self.log("train/value_accuracy", binary_accuracy(y2, result))
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (
            self.hparams.use_ema
            and self.current_epoch >= self.hparams.ema_start_epoch
            and self.global_step % self.hparams.ema_freq == 0
        ):
            self.ema_model.update_parameters(self.model)
        if self.use_swa and self.current_epoch >= self.swa_start_epoch:
            # Update SWA model weights
            self.swa_model.update_parameters(self.model)

    def on_train_epoch_end(self):
        if (
            self.hparams.use_ema
            and self.hparams.update_bn
            and self.current_epoch == self.trainer.max_epochs - 1
            and self.current_epoch >= self.hparams.ema_start_epoch
        ):

            def data_loader():
                for x1, x2, _, _, _ in Tqdm(
                    self.trainer.datamodule.train_dataloader(),
                    desc="update_bn",
                    dynamic_ncols=True,
                    bar_format=TQDMProgressBar.BAR_FORMAT,
                ):
                    yield {"x1": x1.to(self.device), "x2": x2.to(self.device)}

            forward_ = self.ema_model.forward
            self.ema_model.forward = lambda x: forward_(**x)
            with self.trainer.precision_plugin.train_step_context():
                update_bn(data_loader(), self.ema_model)
            del self.ema_model.forward
        if self.use_swa and self.current_epoch >= self.swa_start_epoch:
            self.swa_scheduler.step()

    def on_fit_end(self):
        if self.hparams.model_filename:
            if self.hparams.use_ema:
                model = self.ema_model
            elif self.use_swa:
                # Update batch normalization statistics for SWA model
                dataloader = self.trainer.datamodule.train_dataloader()
                if self.swa_model is not None:
                    update_bn(dataloader, self.swa_model)
                    # Replace the model with the SWA model
                    model = self.swa_model
                else:
                    logger.warning("SWA is enabled but self.swa_model is None. Skipping update_bn and saving base model instead.")
                    model = self.model
            else:
                model = self.model
            model_filename = self.hparams.model_filename.format(
                epoch=self.current_epoch, step=self.global_step
            )
            serializers.save_npz(
                os.path.join(self.trainer.log_dir, model_filename),
                model,
            )

    def validation_step(self, batch, batch_idx):
        features1, features2, move, result, value = batch
        y1, y2 = self.model(features1, features2)
        loss1 = cross_entropy_loss(y1, move).mean()
        loss2 = bce_with_logits_loss(y2, result)
        loss3 = bce_with_logits_loss(y2, value)
        loss = (
            loss1
            + (1 - self.hparams.val_lambda) * loss2
            + self.hparams.val_lambda * loss3
        )
        self.validation_step_outputs["val/loss"].append(loss)
        self.validation_step_outputs["val/policy_loss"].append(loss1)
        self.validation_step_outputs["val/result_loss"].append(loss2)
        self.validation_step_outputs["val/value_loss"].append(loss3)

        self.validation_step_outputs["val/policy_accuracy"].append(accuracy(y1, move))
        self.validation_step_outputs["val/value_accuracy"].append(
            binary_accuracy(y2, result)
        )

        entropy1 = (-F.softmax(y1, dim=1) * F.log_softmax(y1, dim=1)).sum(dim=1)
        self.validation_step_outputs["val/policy_entropy"].append(entropy1.mean())

        p2 = y2.sigmoid()
        # entropy2 = -(p2 * F.log(p2) + (1 - p2) * F.log(1 - p2))
        log1p_ey2 = F.softplus(y2)
        entropy2 = -(p2 * (y2 - log1p_ey2) + (1 - p2) * -log1p_ey2)
        self.validation_step_outputs["val/value_entropy"].append(entropy2.mean())

    def on_validation_epoch_end(self):
        for key, val in self.validation_step_outputs.items():
            self.log(key, torch.stack(val).mean(), sync_dist=True)
            val.clear()

    def on_test_start(self):
        if self.hparams.use_ema:
            self.tmp_model = self.model
            self.model = self.ema_model
        return super().on_test_start()

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        for key, val in self.validation_step_outputs.items():
            key = "test" + key[3:]
            self.log(key, torch.stack(val).mean(), sync_dist=True)
            val.clear()

    def on_test_end(self):
        super().on_test_end()
        if self.hparams.use_ema:
            self.model = self.tmp_model
            del self.tmp_model


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--debug",
            action="store_true",
            default=False,
            help="Debug mode: few batches, 2 epochs, no WandB, small val_check_interval",
        )

    @staticmethod
    def configure_optimizers(lightning_module, optimizer, lr_scheduler=None):
        if lightning_module.hparams.lr_scheduler_interval == "epoch":
            return LightningCLI.configure_optimizers(
                lightning_module, optimizer, lr_scheduler
            )
        if lr_scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                **(
                    {"monitor": lr_scheduler.monitor}
                    if isinstance(lr_scheduler, ReduceLROnPlateau)
                    else {}
                ),
            },
        }

    def before_instantiate_classes(self):
        super().before_instantiate_classes()
        subcommand = self.config.get("subcommand")
        cfg = self.config.get(subcommand) if subcommand else self.config
        if not cfg.get("debug", False):
            return

        logger.info("*** DEBUG MODE: limiting batches and epochs ***")
        cfg["trainer"]["max_epochs"] = 2
        cfg["trainer"]["limit_train_batches"] = 10
        cfg["trainer"]["limit_val_batches"] = 5
        cfg["trainer"]["limit_test_batches"] = 5
        cfg["trainer"]["val_check_interval"] = 5
        cfg["trainer"]["log_every_n_steps"] = 1
        # Replace WandB with a simple CSV logger for debug runs
        cfg["trainer"]["logger"] = {
            "class_path": "lightning.pytorch.loggers.CSVLogger",
            "init_args": {"save_dir": "./lightning_logs"},
        }
        # Disable early stopping patience (not useful for 2 epochs)
        cfg["trainer"]["callbacks"] = [
            cb for cb in (cfg["trainer"].get("callbacks") or [])
            if cb.get("class_path") != "lightning.pytorch.callbacks.EarlyStopping"
        ]


def main():
    CustomLightningCLI(Model, DataModule, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()

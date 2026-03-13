"""FocalModel: subclass of ptl.Model replacing CE/BCE with Focal Loss for all heads.

Policy head: multi-class Focal Loss
  FL(pt) = -alpha * (1 - pt)^gamma * log(pt)

Value head: binary Focal Loss (supports soft targets in [0,1])
  BFL = alpha * (1 - pt)^gamma * BCE,  pt = p*t + (1-p)*(1-t)

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""
import torch
import torch.nn.functional as F

from dlshogi.ptl import Model, accuracy, binary_accuracy, cross_entropy_loss, bce_with_logits_loss


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=1)
    pt = torch.exp(log_p).gather(1, targets.unsqueeze(1)).squeeze(1)
    log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
    return (-alpha * (1.0 - pt) ** gamma * log_pt).mean()


def binary_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    p = torch.sigmoid(logits)
    pt = p * targets + (1.0 - p) * (1.0 - targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (alpha * (1.0 - pt) ** gamma * bce).mean()



class FocalModel(Model):
    """Drops in for Model; uses focal loss for policy and value heads."""

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
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__(
            network=network,
            val_lambda=val_lambda,
            val_lambda_decay_epoch=val_lambda_decay_epoch,
            use_ema=use_ema,
            update_bn=update_bn,
            ema_start_epoch=ema_start_epoch,
            ema_freq=ema_freq,
            ema_decay=ema_decay,
            lr_scheduler_interval=lr_scheduler_interval,
            model_filename=model_filename,
            resume_model=resume_model,
            use_swa=use_swa,
            swa_start_epoch=swa_start_epoch,
            swa_lr=swa_lr,
        )
        self.save_hyperparameters()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def training_step(self, batch, batch_idx):
        features1, features2, move, result, value = batch
        y1, y2 = self.model(features1, features2)
        # CE losses (comparable with other experiments)
        loss1 = cross_entropy_loss(y1, move).mean()
        loss2 = bce_with_logits_loss(y2, result)
        loss3 = bce_with_logits_loss(y2, value)
        loss = (
            loss1
            + (1 - self.hparams.val_lambda) * loss2
            + self.hparams.val_lambda * loss3
        )
        # Focal losses (this experiment's actual training signal)
        focal1 = focal_loss(y1, move, self.focal_alpha, self.focal_gamma)
        focal2 = binary_focal_loss(y2, result, self.focal_alpha, self.focal_gamma)
        focal3 = binary_focal_loss(y2, value, self.focal_alpha, self.focal_gamma)
        focal = (
            focal1
            + (1 - self.hparams.val_lambda) * focal2
            + self.hparams.val_lambda * focal3
        )
        self.log("train/loss", loss)
        self.log("train/policy_loss", loss1)
        self.log("train/result_loss", loss2)
        self.log("train/value_loss", loss3)
        self.log("train/focal_loss", focal)
        self.log("train/focal_policy_loss", focal1)
        self.log("train/focal_result_loss", focal2)
        self.log("train/focal_value_loss", focal3)
        self.log("train/policy_accuracy", accuracy(y1, move))
        self.log("train/value_accuracy", binary_accuracy(y2, result))
        return focal

    def validation_step(self, batch, batch_idx):
        features1, features2, move, result, value = batch
        y1, y2 = self.model(features1, features2)
        # CE losses (comparable with other experiments)
        loss1 = cross_entropy_loss(y1, move).mean()
        loss2 = bce_with_logits_loss(y2, result)
        loss3 = bce_with_logits_loss(y2, value)
        loss = (
            loss1
            + (1 - self.hparams.val_lambda) * loss2
            + self.hparams.val_lambda * loss3
        )
        # Focal losses
        focal1 = focal_loss(y1, move, self.focal_alpha, self.focal_gamma)
        focal2 = binary_focal_loss(y2, result, self.focal_alpha, self.focal_gamma)
        focal3 = binary_focal_loss(y2, value, self.focal_alpha, self.focal_gamma)
        focal = (
            focal1
            + (1 - self.hparams.val_lambda) * focal2
            + self.hparams.val_lambda * focal3
        )
        self.validation_step_outputs["val/loss"].append(loss)
        self.validation_step_outputs["val/policy_loss"].append(loss1)
        self.validation_step_outputs["val/result_loss"].append(loss2)
        self.validation_step_outputs["val/value_loss"].append(loss3)
        self.validation_step_outputs["val/focal_loss"].append(focal)
        self.validation_step_outputs["val/focal_policy_loss"].append(focal1)
        self.validation_step_outputs["val/focal_result_loss"].append(focal2)
        self.validation_step_outputs["val/focal_value_loss"].append(focal3)
        self.validation_step_outputs["val/policy_accuracy"].append(accuracy(y1, move))
        self.validation_step_outputs["val/value_accuracy"].append(binary_accuracy(y2, result))

        entropy1 = (-F.softmax(y1, dim=1) * F.log_softmax(y1, dim=1)).sum(dim=1)
        self.validation_step_outputs["val/policy_entropy"].append(entropy1.mean())

        p2 = y2.sigmoid()
        log1p_ey2 = F.softplus(y2)
        entropy2 = -(p2 * (y2 - log1p_ey2) + (1 - p2) * -log1p_ey2)
        self.validation_step_outputs["val/value_entropy"].append(entropy2.mean())

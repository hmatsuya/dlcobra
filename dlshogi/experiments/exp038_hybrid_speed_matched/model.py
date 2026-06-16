"""exp038: Speed-matched hybrid backbone (dense ResNet + sparse axial-NeXt).

Goal: compare accuracy against the InceptionNeXt baseline (exp026, 86.1M) at
*equal inference speed* rather than equal params. The hybrid is ~1.5x faster
than InceptionNeXt at equal params (see exp037), so a speed-matched hybrid can
be larger.

Speed-match result (exp037/match_speed.py, RTX 3090, torch.compile FP16, batch=256):
  inceptionnext   86.1M  -> 2875 pos/s (baseline)
  hybrid ch480x36 134.2M -> 2799 pos/s  (0.97x, ~matched)

So this experiment uses channels=480, blocks=36, axial_period=5 (134.2M params:
29 dense 3x3 ResNet blocks + 7 axial-NeXt blocks), giving the hybrid ~1.55x more
capacity at the same inference cost as InceptionNeXt.

Activation: GELU is used inside the axial-NeXt MLP regardless; the dense ResNet
blocks use the `activation` passed here (default SiLU/Swish to match the modern
"NeXt"-style baseline). Keep BatchNorm for dense blocks.
"""
import torch.nn as nn

from dlshogi.experiments.exp037_trt_backbone_benchmark.hybrid_model import (
    PolicyValueNetwork as _HybridNetwork,
)


class PolicyValueNetwork(_HybridNetwork):
    def __init__(self, blocks=36, channels=480, axial_period=5, expansion=4,
                 activation=None, fcl=256):
        if activation is None:
            activation = nn.SiLU()
        super().__init__(
            blocks=blocks,
            channels=channels,
            axial_period=axial_period,
            expansion=expansion,
            activation=activation,
            fcl=fcl,
        )

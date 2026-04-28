"""Prunable InceptionNeXt model for exp032.

Identical to exp026 architecture but with configurable MLP expansion ratio,
so the pruned model (with reduced expansion) can be instantiated directly
from a config without modifying the original exp026 model.

Architecture: InceptionNeXt (depths=[40], dims=[512]) + 1 plain self-attention tail.
Default expansion=4 matches exp026/exp029 exactly (86.1M params).
After pruning expansion dim by 25%: expansion_dim = int(512 * 4 * 0.75) = 1536.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM, MAX_MOVE_LABEL_NUM


# ---------------------------------------------------------------------------
# DropPath (Stochastic Depth)
# ---------------------------------------------------------------------------
class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob


# ---------------------------------------------------------------------------
# InceptionNeXt block with configurable expansion_dim
# ---------------------------------------------------------------------------
class InceptionNeXtBlock(nn.Module):
    """InceptionNeXt block with 4 parallel depthwise branches + MLP + DropPath.

    expansion_dim: explicit intermediate dimension for the MLP.
    If None, defaults to dim * expansion (backward-compatible).
    """

    def __init__(self, dim: int, expansion: int = 4, expansion_dim: int | None = None, drop_path: float = 0.0):
        super().__init__()
        assert dim % 4 == 0
        branch_dim = dim // 4
        mlp_dim = expansion_dim if expansion_dim is not None else dim * expansion

        self.dw3x3 = nn.Conv2d(branch_dim, branch_dim, kernel_size=3, padding=1, groups=branch_dim)
        self.dw1x9 = nn.Conv2d(branch_dim, branch_dim, kernel_size=(1, 9), padding=(0, 4), groups=branch_dim)
        self.dw9x1 = nn.Conv2d(branch_dim, branch_dim, kernel_size=(9, 1), padding=(4, 0), groups=branch_dim)

        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, mlp_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mlp_dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        b = x.shape[1] // 4

        x0, x1, x2, x3 = x[:, :b], x[:, b:2*b], x[:, 2*b:3*b], x[:, 3*b:]
        x1 = self.dw3x3(x1)
        x2 = self.dw1x9(x2)
        x3 = self.dw9x1(x3)

        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        return residual + self.drop_path(x)


# ---------------------------------------------------------------------------
# Plain self-attention block (unchanged from exp026)
# ---------------------------------------------------------------------------
class PlainSelfAttentionBlock(nn.Module):
    """Minimal transformer block: MHA + FFN, pre-norm, no positional encoding, with DropPath.

    Input/output: (B, C, H, W). Internally flattens to (B, H*W, C).
    """

    def __init__(self, dim: int, num_heads: int = 8, ff_expansion: int = 4, drop_path: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.d_head = dim // num_heads
        self.scale = self.d_head ** -0.5

        # Attention
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

        # FFN
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_expansion),
            nn.GELU(),
            nn.Linear(dim * ff_expansion, dim),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 81, C)

        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, self.d_head)
        q, k, v = qkv.unbind(2)  # each (B, 81, heads, d_head)
        q = q.transpose(1, 2)  # (B, heads, 81, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        x = self.out_proj(x)
        x = residual + self.drop_path(x)

        # FFN with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.drop_path(x)

        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------
class PolicyValueNetwork(nn.Module):
    def __init__(
        self,
        depths: tuple = (40,),
        dims: tuple = (512,),
        num_heads: int = 8,
        ff_expansion: int = 4,
        drop_path_rate: float = 0.2,
        fcl: int = 256,
        # Pruning parameter: explicit MLP intermediate dim per block.
        # None = use dims[0] * 4 (default, matches exp026 exactly).
        # Set to e.g. 1536 for 25%-pruned model (512*4*0.75).
        mlp_expansion_dim: int | None = None,
    ):
        super().__init__()
        dim = dims[0]
        num_inception = depths[0] - 1  # 39 InceptionNeXt + 1 attention
        total_blocks = depths[0]

        self.stem_1 = nn.Conv2d(FEATURES1_NUM, dim, kernel_size=3, padding=1, bias=False)
        self.stem_2 = nn.Conv2d(FEATURES2_NUM, dim, kernel_size=1, bias=False)
        self.stem_norm = nn.LayerNorm(dim)

        # Linear drop path schedule across all blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.blocks = nn.Sequential(
            *[
                InceptionNeXtBlock(dim, expansion_dim=mlp_expansion_dim, drop_path=dpr[i])
                for i in range(num_inception)
            ],
            PlainSelfAttentionBlock(dim, num_heads=num_heads, ff_expansion=ff_expansion, drop_path=dpr[-1]),
        )

        self.policy = nn.Conv2d(dim, MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.policy_bias = nn.Parameter(torch.zeros(9 * 9 * MAX_MOVE_LABEL_NUM))

        self.value_conv = nn.Conv2d(dim, MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.value_norm = nn.LayerNorm(MAX_MOVE_LABEL_NUM)
        self.value_fc1 = nn.Linear(9 * 9 * MAX_MOVE_LABEL_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)
        self.act = nn.GELU()

    def forward(self, x1, x2):
        x = self.stem_1(x1) + self.stem_2(x2)
        x = x.permute(0, 2, 3, 1)
        x = self.stem_norm(x)
        x = x.permute(0, 3, 1, 2)

        x = self.blocks(x)

        h_policy = self.policy(x)
        h_policy = torch.flatten(h_policy, 1) + self.policy_bias

        h_value = self.value_conv(x)
        h_value = h_value.permute(0, 2, 3, 1)
        h_value = self.act(self.value_norm(h_value))
        h_value = h_value.permute(0, 3, 1, 2)
        h_value = self.act(self.value_fc1(torch.flatten(h_value, 1)))
        h_value = self.value_fc2(h_value)

        return h_policy, h_value

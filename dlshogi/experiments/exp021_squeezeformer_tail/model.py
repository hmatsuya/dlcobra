"""InceptionNeXt + Squeezeformer tail block for Shogi.

Based on exp015 (depths=[10], dims=[192]).
Replaces the last InceptionNeXt block with a Squeezeformer block
adapted for 2D board input (9x9 flattened to seq_len=81).

Squeezeformer block structure (NeurIPS 2022):
  MHA + LayerNorm -> FFN + LayerNorm -> ConvModule + LayerNorm -> FFN + LayerNorm
All with residual connections. Simplified from the original speech model:
- No time reduction/recovery (fixed 9x9 board)
- Relative positional encoding for spatial awareness on the board
- 1D convolution operates on flattened board sequence (length 81)

Reference: https://github.com/upskyy/Squeezeformer
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dlshogi.common import *


# ---------------------------------------------------------------------------
# InceptionNeXt block (same as exp015)
# ---------------------------------------------------------------------------
class InceptionNeXtBlock(nn.Module):
    """InceptionNeXt block with 4 parallel depthwise branches + MLP."""

    def __init__(self, dim, expansion=4):
        super().__init__()
        assert dim % 4 == 0
        branch_dim = dim // 4

        self.dw3x3 = nn.Conv2d(branch_dim, branch_dim, kernel_size=3, padding=1, groups=branch_dim)
        self.dw1x9 = nn.Conv2d(branch_dim, branch_dim, kernel_size=(1, 9), padding=(0, 4), groups=branch_dim)
        self.dw9x1 = nn.Conv2d(branch_dim, branch_dim, kernel_size=(9, 1), padding=(4, 0), groups=branch_dim)

        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * expansion, dim)

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

        return x + residual


# ---------------------------------------------------------------------------
# Squeezeformer components (adapted from upskyy/Squeezeformer for 2D board)
# ---------------------------------------------------------------------------

class RelPositionalEncoding(nn.Module):
    """Relative positional encoding for board sequences."""

    def __init__(self, d_model, max_len=128):
        super().__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x):
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb


class RelativeMultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional encoding."""

    def __init__(self, d_model, num_heads, dropout_p=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(num_heads, self.d_head))
        nn.init.xavier_uniform_(self.u_bias)
        nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, pos_embedding):
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim
        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score):
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded = torch.cat([zeros, pos_score], dim=-1)
        padded = padded.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]
        return pos_score


class MultiHeadedSelfAttentionModule(nn.Module):
    """Self-attention with relative positional encoding."""

    def __init__(self, d_model, num_heads, dropout_p=0.0):
        super().__init__()
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding)
        return self.dropout(outputs)


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        outputs, gate = x.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class Transpose(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)


class FeedForwardModule(nn.Module):
    """Squeezeformer FFN: Linear -> Swish -> Dropout -> Linear -> Dropout."""

    def __init__(self, encoder_dim, expansion_factor=4, dropout_p=0.0):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs):
        return self.sequential(inputs)


class ConvModule(nn.Module):
    """Squeezeformer convolution module: Pointwise -> GLU -> DepthwiseConv1d -> BN -> Swish -> Pointwise."""

    def __init__(self, in_channels, kernel_size=9, dropout_p=0.0):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_channels, in_channels * 2, kernel_size=1, bias=True),
            GLU(dim=1),
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                      padding=(kernel_size - 1) // 2, groups=in_channels),
            nn.BatchNorm1d(in_channels),
            Swish(),
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs):
        return self.sequential(inputs).transpose(1, 2)


class ResidualConnectionModule(nn.Module):
    """Residual: output = module(input) * factor + input."""

    def __init__(self, module, module_factor=1.0):
        super().__init__()
        self.module = module
        self.module_factor = module_factor

    def forward(self, inputs):
        return self.module(inputs) * self.module_factor + inputs


class SqueezeformerBlock(nn.Module):
    """Squeezeformer block adapted for 2D board (operates on flattened seq_len=81).

    Structure: MHA+LN -> FFN+LN -> Conv+LN -> FFN+LN (all with residual).
    Input/output: (B, C, H, W) - converts internally to (B, H*W, C) for attention.
    """

    def __init__(self, dim, num_heads=8, ff_expansion=4, conv_kernel_size=9, dropout_p=0.0):
        super().__init__()
        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                MultiHeadedSelfAttentionModule(d_model=dim, num_heads=num_heads, dropout_p=dropout_p),
            ),
            nn.LayerNorm(dim),
            ResidualConnectionModule(
                FeedForwardModule(encoder_dim=dim, expansion_factor=ff_expansion, dropout_p=dropout_p),
            ),
            nn.LayerNorm(dim),
            ResidualConnectionModule(
                ConvModule(in_channels=dim, kernel_size=conv_kernel_size, dropout_p=dropout_p),
            ),
            nn.LayerNorm(dim),
            ResidualConnectionModule(
                FeedForwardModule(encoder_dim=dim, expansion_factor=ff_expansion, dropout_p=dropout_p),
            ),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        # x: (B, C, H, W) -> flatten to (B, H*W, C) for transformer ops
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 81, C)
        x = self.sequential(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        return x


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------
class PolicyValueNetwork(nn.Module):
    def __init__(
        self,
        depths=(10,),
        dims=(192,),
        num_heads=8,
        ff_expansion=4,
        conv_kernel_size=9,
        fcl=256,
    ):
        super().__init__()
        dim = dims[0]
        num_inception = depths[0] - 1  # 9 InceptionNeXt + 1 Squeezeformer

        self.stem_1 = nn.Conv2d(FEATURES1_NUM, dim, kernel_size=3, padding=1, bias=False)
        self.stem_2 = nn.Conv2d(FEATURES2_NUM, dim, kernel_size=1, bias=False)
        self.stem_norm = nn.LayerNorm(dim)

        self.blocks = nn.Sequential(
            *[InceptionNeXtBlock(dim) for _ in range(num_inception)],
            SqueezeformerBlock(dim, num_heads=num_heads, ff_expansion=ff_expansion,
                               conv_kernel_size=conv_kernel_size),
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

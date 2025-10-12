# unet/attention_modules.py
"""
Attention modules for U-Net
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class KNNAttention(nn.Module):
    """
    K-NN Sparse Self-Attention (Adaptive-k version)
    ------------------------------------------------
    Each query predicts τ ∈ (0,1) based on its features to dynamically determine k = τ·k_max.
    Backward compatible: if set adaptive=False, degrades to fixed k.
    """
    def __init__(
        self,
        dim: int,
        k_max: int = 32,
        k_min: int = 4,
        heads: int = 4,
        adaptive: bool = True,
        bias: bool = False,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % heads == 0
        self.dim, self.h = dim, heads
        self.dh = dim // heads
        self.k_max, self.k_min = k_max, k_min
        self.adaptive = adaptive

        # QKV projection
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        # Small gate module, use 1×1 Conv to predict τ
        if adaptive:
            self.gate = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 1), 
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 4, 1, 1),   
                nn.Sigmoid()
            )
            nn.init.constant_(self.gate[-2].bias, -1.0)   # Initial bias towards small k

    def _reshape(self, t: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w = t.shape
        t = t.reshape(b, self.h, self.dh, h * w).permute(0, 1, 3, 2)
        return t, h * w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        q, N = self._reshape(q)
        k, _ = self._reshape(k)
        v, _ = self._reshape(v)

        sim = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.dh)

        # Dynamic k based on feature map size
        if self.adaptive:
            τ = self.gate(x)                           # [B,1,H,W]
            τ = τ.reshape(B, 1, N)                    # align to tokens
            k_dyn = (τ * self.k_max).long().clamp(self.k_min, self.k_max)  # [B,1,N]
        else:
            k_dyn = torch.full((B, 1, N), self.k_max, device=x.device, dtype=torch.long)

        # Take k_max and mask
        topk_val, topk_idx = torch.topk(sim, self.k_max, dim=-1)            # [B,h,N,k_max]
        mask = (
            torch.arange(self.k_max, device=x.device)
            .view(1, 1, 1, -1)
            < k_dyn.unsqueeze(-1)               # broadcast
        )
        topk_val = topk_val.masked_fill(~mask, float("-inf"))               # Mask logits
        attn = F.softmax(topk_val, dim=-1)                                   # [B,h,N,k_max]

        # gather V
        # gather V
        v_flat = v.reshape(B * self.h, N, self.dh)
        g_idx  = topk_idx.reshape(B * self.h, N, self.k_max)

        # Fix: expand v_flat to 4D before gather
        v_expanded = v_flat.unsqueeze(2).expand(-1, -1, self.k_max, -1)  # [B*h, N, k_max, dh]
        g_idx_expanded = g_idx.unsqueeze(-1).expand(-1, -1, -1, self.dh)  # [B*h, N, k_max, dh]
        gathered = torch.gather(v_expanded, 1, g_idx_expanded)            # [B*h, N, k_max, dh]
        gathered = gathered.reshape(B, self.h, N, self.k_max, self.dh)    # [B, h, N, k_max, dh]
        out = torch.sum(attn.unsqueeze(-1) * gathered, dim=-2)               # [B,h,N,dh]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        out = self.proj_drop(self.proj(out))
        return x + out


def get_dynamic_k_max(h: int, w: int, base_k: int = 32) -> int:
    """
    Dynamically adjust k_max based on feature map size
    Smaller feature maps get smaller k values for efficiency
    """
    n_tokens = h * w
    if n_tokens <= 256:      # 16x16
        return min(base_k // 4, n_tokens // 2)
    elif n_tokens <= 1024:   # 32x32
        return min(base_k // 2, n_tokens // 4)
    elif n_tokens <= 4096:   # 64x64
        return min(base_k, n_tokens // 8)
    else:                    # larger
        return min(base_k, n_tokens // 16)
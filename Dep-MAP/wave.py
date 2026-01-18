# =========================================================
# Wavelet Fusion (inlined from wave.py for public release)
# Dependency: PyWavelets (pywt)
# =========================================================

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = F.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = F.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = F.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = F.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = F.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = F.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = F.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = F.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = F.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = F.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT_2D(nn.Module):
    def __init__(self, wave: str):
        super().__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32)
        rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)

        self.register_buffer("filters", filters)  # (4,1,k,k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return IDWT_Function.apply(x, self.filters)


class DWT_2D(nn.Module):
    def __init__(self, wave: str):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer("w_ll", w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer("w_lh", w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer("w_hl", w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer("w_hh", w_hh.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class _FusionResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.conv1(x))
        out = F.gelu(self.conv2(out))
        return out + x


class Fusion(nn.Module):
    """
    Wavelet Top-Down Fusion block.

    Inputs:
        x1: (B, C, H, W)  higher-res feature
        x2: (B, C, H/2, W/2) (typically) lower-res feature
    Output:
        out: (B, C, H, W)
    """
    def __init__(self, in_channels: int, wave: str = "haar"):
        super().__init__()
        self.dwt = DWT_2D(wave)
        self.convh1 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.high = _FusionResBlock(in_channels)
        self.convh2 = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, stride=1, padding=0, bias=True)

        self.convl = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.low = _FusionResBlock(in_channels)

        self.idwt = IDWT_2D(wave)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x1.shape

        x_dwt = self.dwt(x1)
        ll, lh, hl, hh = x_dwt.split(c, dim=1)

        high = torch.cat([lh, hl, hh], dim=1)         # (B,3C,H/2,W/2)
        high = self.convh1(high)                      # (B,C,H/2,W/2)
        high = self.high(high)                        # (B,C,H/2,W/2)
        high = self.convh2(high)                      # (B,3C,H/2,W/2)

        # Align low-res x2 to ll if needed
        if ll.shape[-2:] != x2.shape[-2:]:
            # keep original behavior: pad top by 1 when H mismatch
            # (this matches the original wave.py)
            if ll.shape[-2] != x2.shape[-2]:
                x2 = F.pad(x2, (0, 0, 1, 0), "constant", 0)

        low = torch.cat([ll, x2], dim=1)              # (B,2C,H/2,W/2)
        low = self.convl(low)                         # (B,C,H/2,W/2)
        low = self.low(low)                           # (B,C,H/2,W/2)

        out = torch.cat([low, high], dim=1)           # (B,4C,H/2,W/2)
        out = self.idwt(out)                          # (B,C,H,W)
        return out

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torchvision import models
from einops import rearrange

from attention import TransformerBlock, guide_attention
from wave import Fusion

def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update

def sinkhorn_knopp(out, n_iterations=3, epsilon=0.05, use_gumbel=False):
    L = torch.exp(out / epsilon).t()  # shape: [K, B,]
    K, B = L.shape

    # make the matrix sums to 1
    sum_L = torch.sum(L)
    L /= sum_L

    for _ in range(n_iterations):
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K

        L /= torch.sum(L, dim=0, keepdim=True)
        L /= B

    L *= B
    L = L.t()

    indices = torch.argmax(L, dim=1)
    if use_gumbel:
        L = F.gumbel_softmax(L, tau=0.5, hard=True)
    else:
        L = F.one_hot(indices, num_classes=K).to(dtype=torch.float32)

    return L, indices

def masked_mean(
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim,
    keepdim: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    x: Tensor(...)
    mask: broadcastable to x on dim(s) being reduced; dtype can be bool/0-1
    dim: int or tuple[int]
    """
    if mask is None:
        return x.mean(dim=dim, keepdim=keepdim)
    m = mask.to(dtype=x.dtype)
    num = (x * m).sum(dim=dim, keepdim=keepdim)
    den = m.sum(dim=dim, keepdim=keepdim).clamp(min=eps)
    return num / den

def feature_masked_mean(
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    keepdim: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    x: (B, T, C, H, W)
    mask: (B, T) or None
    """
    if mask is None:
        global_mean = x.mean(dim=(0, 1, 2))              # (H, W)
        xmean = x.mean(dim=2, keepdim=keepdim)              # (B, T, 1, H, W)
    else:
        m = mask[:, :, None, None, None].float()         # (B, T, 1, 1, 1)
        x_m = x * m
        valid = m.sum().clamp_min(eps)

        global_mean = x_m.sum(dim=(0, 1, 2)) / valid     # (H, W)
        xmean = x_m.mean(dim=2, keepdim=keepdim)            # (B, T, 1, H, W)

    return xmean, global_mean

# ---------------------------
# Baseline
# ---------------------------
@dataclass
class ResNet18Output:
    x1: torch.Tensor  # (B*T, 64,  H/4,  W/4)
    x2: torch.Tensor  # (B*T, 128, H/8,  W/8)
    x3: torch.Tensor  # (B*T, 256, H/16, W/16)
    x4: torch.Tensor  # (B*T, 512, H/32, W/32)


PretrainedMode = Optional[Literal["imagenet", "hf_emotion"]]


def _build_resnet18(pretrained: PretrainedMode) -> nn.Module:
    """
    Build a torchvision ResNet-18.
    - imagenet: load official torchvision weights
    - hf_emotion: load HF fine-tuned weights (ignore fc.*)
    - None: random init
    """
    if pretrained == "imagenet":
        # New torchvision API (recommended)
        resnet = models.resnet18(pretrained=True)
        return resnet

    # Otherwise, start from random init
    resnet = models.resnet18(weights=None)

    if pretrained == "hf_emotion":
        # Lazy import to avoid dependency if not used
        from huggingface_hub import hf_hub_download

        weight_path = hf_hub_download(
            repo_id="mark1316/best_emotion_detection",
            filename="pytorch_model.bin",
        )
        state = torch.load(weight_path, map_location="cpu")

        # HF state_dict contains fc.* for 8-class head, but we only need backbone features.
        # Drop fc to avoid shape mismatch / unnecessary head.
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}

        # strict=False because we intentionally dropped fc.*
        missing, unexpected = resnet.load_state_dict(state, strict=False)

        # Optional sanity check (you can comment out if you want ultra-minimal)
        # We expect missing keys only from "fc.weight" and "fc.bias"
        # and no unexpected keys.
        if any(k for k in missing if not k.startswith("fc.")):
            raise RuntimeError(f"Unexpected missing keys (non-fc): {missing}")
        if len(unexpected) > 0:
            raise RuntimeError(f"Unexpected keys in HF state_dict: {unexpected}")

    return resnet


class ResNet18MultiScale(nn.Module):
    """
    Return feature maps at stages:
      x1: 64  @ 1/4  (224x224 -> 56x56)
      x2: 128 @ 1/8  (28x28)
      x3: 256 @ 1/16 (14x14)
      x4: 512 @ 1/32 (7x7)

    Notes:
    - This module ONLY returns multi-scale feature maps (no avgpool/fc).
    - If pretrained="hf_emotion", it loads the HF fine-tuned weights and drops fc.*.
    """
    def __init__(self, pretrained: PretrainedMode = "hf_emotion"):
        super().__init__()
        resnet = _build_resnet18(pretrained)

        # Keep the exact same feature extraction path as torchvision ResNet:
        # conv1 -> bn1 -> relu -> maxpool  (output stride /4)
        self.stem = nn.Sequential(*list(resnet.children())[0:4])

        # Residual stages
        self.layer1 = nn.Sequential(*list(resnet.children())[4:5])
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6])
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7])
        self.layer4 = nn.Sequential(*list(resnet.children())[7:8])

    def forward(self, x: torch.Tensor) -> ResNet18Output:
        # x: (B*T, 3, H, W)
        x = self.stem(x)       # (B*T, 64, H/4,  W/4)
        x1 = self.layer1(x)    # (B*T, 64, H/4,  W/4)
        x2 = self.layer2(x1)   # (B*T, 128,H/8,  W/8)
        x3 = self.layer3(x2)   # (B*T, 256,H/16, W/16)
        x4 = self.layer4(x3)   # (B*T, 512,H/32, W/32)
        return ResNet18Output(x1=x1, x2=x2, x3=x3, x4=x4)

# ---------------------------
# AU-Semantic Guide Attention
# ---------------------------
@dataclass
class AUSGAOutput:
    """
    Output of AU-SGA.
    All tensors keep the same shape as backbone feature maps.
    """
    x1_au: torch.Tensor   # (B*T, 64, 56, 56)
    x2_au: torch.Tensor   # (B*T,128, 28, 28)
    x3_au: torch.Tensor   # (B*T,256, 14, 14)
    x4_au: torch.Tensor   # (B*T,512,  7,  7)

class AUSGA(nn.Module):
    """
    AU Semantic-Guided Attention (AU-SGA)

    Use CLIP encoder outputs (NO projection layer) as semantic guidance
    to enhance multi-scale CNN feature maps via guide_attention.
    """

    def __init__(self):
        super().__init__()

        # ResNet18 @ 224x224 feature specs
        # (channels, spatial_size)
        self.scales = [
            (64, 56),
            (128, 28),
            (256, 14),
            (512, 7),
        ]

        text_dim = 512   # CLIP text encoder width (before projection)
        img_dim = 768    # CLIP image encoder width (before projection)

        self.text_proj = nn.ModuleList()
        self.img_proj = nn.ModuleList()
        self.attn = nn.ModuleList()

        for c, hw in self.scales:
            # text semantic projection
            self.text_proj.append(
                nn.Sequential(
                    nn.LayerNorm(text_dim),
                    nn.Linear(text_dim, c),
                )
            )

            # image semantic projection
            self.img_proj.append(
                nn.Sequential(
                    nn.LayerNorm(img_dim),
                    nn.Linear(img_dim, c),
                )
            )

            # semantic-guided attention (existing module)
            self.attn.append(
                guide_attention(img_c=c, img_hw=hw)
            )

    def forward(
        self,
        text_sem: torch.Tensor,   # (B*T, 512)
        img_sem: torch.Tensor,    # (B*T, 768)
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
    ) -> AUSGAOutput:
        """
        Forward AU-SGA.

        Args:
            text_sem: CLIP text encoder output, no projection (B*T, 512)
            img_sem : CLIP image encoder output, no projection (B*T, 768)
            x1..x4  : multi-scale CNN feature maps

        Returns:
            AUSGAOutput with guided feature maps.
        """
        xs = [x1, x2, x3, x4]
        outs = []

        for i, (x, (c, hw)) in enumerate(zip(xs, self.scales)):
            BT = x.shape[0]

            # build semantic maps: (B*T, C, H*W)
            t_map = self.text_proj[i](text_sem).view(BT, c, -1)
            i_map = self.img_proj[i](img_sem).view(BT, c, -1)

            # semantic-guided attention
            x_au = self.attn[i](t_map, i_map, x)
            outs.append(x_au)

        return AUSGAOutput(
            x1_au=outs[0],
            x2_au=outs[1],
            x3_au=outs[2],
            x4_au=outs[3],
        )

# ---------------------------
# Multi-Scale Representation Module
# ---------------------------
@dataclass
class MSROutput:
    fmap: torch.Tensor       # (B, T, 128, H1, W1) typically (B,T,128,32,32) for 128 input
    frame_emb: torch.Tensor  # (B, T, 128)
    frame_predict: torch.Tensor  # (B, T, 1)
    x1_attn: torch.Tensor   # (B*T, 64, 56, 56)
    x2_attn: torch.Tensor   # (B*T,128, 28, 28)
    x3_attn: torch.Tensor   # (B*T,256, 14, 14)
    x4_attn: torch.Tensor   # (B*T,512,  7,  7)

class MSR(nn.Module):
    """
    MSR module (multi-scale refine + wavelet top-down fusion)

    Expected inputs:
        x1: (B*T,  64, H1, W1)
        x2: (B*T, 128, H2, W2)
        x3: (B*T, 256, H3, W3)
        x4: (B*T, 512, H4, W4)
      where typically: H1>H2>H3>H4, W1>W2>W3>W4

    Args passed to forward:
        b: original batch size B
        t: original time length T
        mask: Optional[(B, T)] with 1 for valid frame, 0 for padding/invalid

    Outputs:
        fmap         : (B, T, mid_dim, Hf, Wf)
        frame_emb    : (B, T, mid_dim)
        frame_predict: (B, T, 1)
        x_attn 1-4       : (BT, dim, h' ,w')
    """
    def __init__(self, mid_dim: int = 64, wave: str = "haar"):
        super().__init__()

        # project to mid_dim
        self.proj1 = nn.Conv2d(64,  mid_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(128, mid_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(256, mid_dim, kernel_size=1)
        self.proj4 = nn.Conv2d(512, mid_dim, kernel_size=1)
        self.activation = nn.GELU()

        # refine (attention.py TransformerBlock expects (B,C,H,W))
        self.refine1 = TransformerBlock(mid_dim, 8, 2.66, False, "WithBias")
        self.refine2 = TransformerBlock(mid_dim, 8, 2.66, False, "WithBias")
        self.refine3 = TransformerBlock(mid_dim, 8, 2.66, False, "WithBias")
        self.refine4 = TransformerBlock(mid_dim, 8, 2.66, False, "WithBias")

        # wavelet fusion (top-down)
        self.fuse34 = Fusion(mid_dim, wave)
        self.fuse23 = Fusion(mid_dim, wave)
        self.fuse12 = Fusion(mid_dim, wave)

        self.post = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Linear(1024, 1)

        self.register_buffer("global_mean", torch.zeros(1, 1, 1))  # place

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        *,
        b: int,
        t: int,
        mask: Optional[torch.Tensor] = None,
    ) -> MSROutput:
        """
        x1..x4: (B*T, C, H, W)
        b,t: original batch and time, used to reshape back
        mask: (B,T) 1/0
        """
        # align channels
        x1 = self.refine1(self.activation(self.proj1(x1)))  # (B*T,128,56,56)
        x2 = self.refine2(self.activation(self.proj2(x2)))  # (B*T,128,28,28)
        x3 = self.refine3(self.activation(self.proj3(x3)))  # (B*T,128,14,14)
        x4 = self.refine4(self.activation(self.proj4(x4)))  # (B*T,128, 7, 7)

        # fuse top-down: output resolution follows the first arg (higher-res)
        x34 = self.fuse34(x3, x4)  # -> ~14x14
        x23 = self.fuse23(x2, x34) # -> ~28x28
        x12 = self.fuse12(x1, x23) # -> ~56x56

        x_fused = self.post(x12)   # (B*T,128,32,32)

        # reshape back to (B,T,128,H,W)
        fmap = rearrange(x_fused, "(b t) c h w -> b t c h w", b=b, t=t)

        xmean, global_mean = feature_masked_mean(fmap, mask=mask)

        if self.global_mean.shape != global_mean.shape:
            self.global_mean.resize_as_(global_mean).copy_(global_mean.detach())  # (H,W)
        elif self.training:
            self.global_mean.mul_(0.99).add_(global_mean.detach(), alpha=0.01)
            
        frame_emb = self.dropout(torch.flatten(xmean + (self.global_mean + global_mean) / 2, start_dim=2))
        # frame_emb = self.dropout(torch.flatten(feature_masked_mean(fmap, mask=mask), start_dim=2))

        frame_predict = self.classifier(frame_emb)

        return MSROutput(fmap=fmap, frame_emb=frame_emb, frame_predict=frame_predict,x1_attn=x1, x2_attn=x2, x3_attn=x3, x4_attn=x4)

# ---------------------------
# PCA foreground mask
# ---------------------------
@dataclass
class PCAMaskOutput:
    mask_bhw: torch.Tensor     # (BT, H, W) bool
    scores_bhw: torch.Tensor   # (BT, H, W) float

class PCAMask(nn.Module):
    """
    PCA-based foreground (face) region detection on spatial feature maps.

    Input:
        x: (BT, C, H, W)
           C is feature dim (old D), N = H * W
        valid_mask: (BT,) 0/1 or bool
           if 0 => return all-zero facemask & scores
    Output:
        mask_bhw: (BT, H, W) bool
        scores_bhw: (BT, H, W) float
    """

    def __init__(
        self,
        keep_quantile: float = 0.6,
        niter: int = 5,
        normalize_scores: bool = True,
    ):
        super().__init__()
        assert 0.0 < keep_quantile < 1.0
        self.keep_quantile = keep_quantile
        self.niter = niter
        self.normalize_scores = normalize_scores

    def forward(
        self,
        x: torch.Tensor,           # (BT, C, H, W)
        valid_mask: torch.Tensor,  # (BT,) 0/1 or bool
    ) -> PCAMaskOutput:

        BT, C, H, W = x.shape
        device = x.device

        if valid_mask.shape != (BT,):
            raise ValueError(f"valid_mask must be (BT,)={(BT,)}, got {valid_mask.shape}")
        valid_mask = valid_mask.bool()

        mask_bhw = torch.zeros(BT, H, W, dtype=torch.bool, device=device)
        scores_bhw = torch.zeros(BT, H, W, dtype=x.dtype, device=device)

        # per-sample PCA (per frame)
        for i in range(BT):
            if not valid_mask[i]:
                continue  # keep zeros

            # (C, H, W) -> (N, C)
            xi = x[i].flatten(1).transpose(0, 1)  # (H*W, C)

            # PCA (q=1)
            U, S, V = torch.pca_lowrank(xi, q=1, center=True, niter=self.niter)

            si = U[:, 0] * S[0]  # (N,)

            # normalize to [0,1] for stable quantile
            if self.normalize_scores:
                s_min = si.min()
                s_max = si.max()
                si = (si - s_min) / (s_max - s_min + 1e-6)

            thr = torch.quantile(si, self.keep_quantile)
            mi = si >= thr  # (N,) bool

            mask_bhw[i] = mi.view(H, W)
            scores_bhw[i] = si.view(H, W)

        return PCAMaskOutput(mask_bhw=mask_bhw, scores_bhw=scores_bhw)

# ---------------------------
# Hash coding on foreground only (BTCHW)
# ---------------------------
@dataclass
class HashCodeOutput:
    code: torch.Tensor      # (BT, C, 2*topk)


def hash_coding_foreground(
    feat: torch.Tensor,     # (BT, C, H, W)
    pool_size: int = 8,
    topk: int = 8,
) -> HashCodeOutput:
    """
    Foreground hash coding without PCAMask / mask_hw.
    Foreground locations are selected purely by feature activation strength.
    """

    BT, C, H, W = feat.shape

    # ------------------------------------------------
    # 1) spatial pooling: (H,W) -> (p,p)
    # ------------------------------------------------
    pooled = F.adaptive_avg_pool2d(feat, (pool_size, pool_size))
    # pooled: (BT, C, p, p)

    # ------------------------------------------------
    # 2) flatten spatial grid
    # ------------------------------------------------
    flat = pooled.view(BT, C, -1)   # (BT, C, p*p)

    # ------------------------------------------------
    # 3) top-k spatial locations by activation
    # ------------------------------------------------
    _, idx = torch.topk(
        flat,
        k=topk,
        dim=-1,
        largest=True,
        sorted=True
    )  # (BT, C, topk)

    # ------------------------------------------------
    # 4) encode (row, col)
    # ------------------------------------------------
    row = idx // pool_size
    col = idx % pool_size

    code = torch.cat(
        [row.float(), col.float()],
        dim=-1
    )  # (BT, C, 2*topk)

    return HashCodeOutput(code=code)

# ---------------------------
# (A) Semantic Disambiguation Prototype Cluster (SDPC)
# ---------------------------
@dataclass
class SDPCOutput:
    patch_logits: torch.Tensor        # (BT, C, CLASS+1, K)
    part_assignments: torch.Tensor    # (BT, C)
    tw: torch.Tensor                 # (B, T)
    topk_per_sample: torch.Tensor

class SDPC(nn.Module):
    """
    Semantic Disambiguation Prototype Cluster (SDPC)

    Inputs:
        hash_feat    : (BT, C, hash_dim)
        tokens       : (BT, C, feature_dim)
        labels       : (B,)          # video-level class label
        mask         : (B, T)        # 1 is valid

    Key ideas:
        - Sparse measurement along time axis (T) for each channel c
        - Only (t,c) in M_top keep original labels as important frames
        - All others assigned to an explicit "Other" class
        - All (t,c) participate in online prototype clustering
    """

    def __init__(
        self,
        *,
        num_classes: int,
        num_prototypes: int,
        hash_dim: int,
        feature_dim: int,
        c: int,
        gamma: float = 0.999,
        use_sinkhorn: bool = True,
        use_gumbel: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.other_class_id = num_classes
        self.total_classes = num_classes + 1

        self.num_prototypes = num_prototypes
        self.feature_dim = feature_dim

        self.c = c
        self.gamma = gamma
        self.use_sinkhorn = use_sinkhorn
        self.use_gumbel = use_gumbel

        self.embed = nn.Linear(hash_dim, feature_dim)

        # prototypes: (Class+1, K, D)
        self.register_buffer(
            "prototypes",
            torch.randn(self.total_classes, num_prototypes, feature_dim)
        )
        nn.init.trunc_normal_(self.prototypes, std=0.02)

    # -------------------------------------------------
    # Sparse measurement along time axis
    # -------------------------------------------------
    def _prob_QK(
            self,
            Q: torch.Tensor,  # (B, C, T, D)
            K: torch.Tensor,  # (B, C, T, D)
            *,
            mask: torch.Tensor,  # (B, T) bool, 1 is valid
            c_factor: float,  # self.c
    ):
        """
        Sparse measurement along time axis with variable valid length.

        Returns:
            M_top_idx: (B, C, u_q_max)  # padded with valid indices if needed
        """
        B, C, T, D = Q.shape
        device = Q.device

        mask = mask.bool()
        valid_lens = mask.long().sum(dim=1)  # (B,)

        # compute per-sample u_k/u_q, then take max for padding
        u_k_list = []
        u_q_list = []
        for b in range(B):
            Lb = int(valid_lens[b].item())
            Lb = max(Lb, 1)
            logT = math.log(max(Lb, 2))
            uk = max(1, min(int(c_factor * logT), Lb))
            uq = max(1, min(int(c_factor * logT), Lb))
            u_k_list.append(uk)
            u_q_list.append(uq)

        u_k_max = max(u_k_list)
        u_q_max = max(u_q_list)

        M_top_idx = torch.zeros(B, C, u_q_max, device=device, dtype=torch.long)

        # ---- per-sample loop: correct + simple for variable length ----
        for b in range(B):
            valid_idx = torch.nonzero(mask[b], as_tuple=False).squeeze(1)  # (Lb,)
            Lb = valid_idx.numel()
            if Lb == 0:
                # fallback: no valid frame, keep zeros
                continue

            uk = u_k_list[b]
            uq = u_q_list[b]

            # gather valid Q/K: (C, Lb, D)
            Qb = Q[b, :, valid_idx, :]  # (C, Lb, D)
            Kb = K[b, :, valid_idx, :]  # (C, Lb, D)

            # sample indices within [0, Lb)
            # for each query time t' in [0,Lb), sample uk keys
            index_sample = torch.randint(0, Lb, (Lb, uk), device=device)  # (Lb, uk)

            # K_sample: (C, Lb, uk, D)
            Kb_expand = Kb.unsqueeze(1).expand(C, Lb, Lb, D)  # (C, Lb, Lb, D)
            K_sample = Kb_expand[:, torch.arange(Lb, device=device).unsqueeze(1), index_sample, :]  # (C,Lb,uk,D)

            # QK: (C, Lb, uk)
            QK = torch.matmul(Qb.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

            # M: (C, Lb)
            M = QK.max(dim=-1).values - QK.mean(dim=-1)

            # topk on valid positions only: get indices in [0,Lb)
            top_local = torch.topk(M, k=uq, dim=-1).indices  # (C, uq)

            # map local idx -> global time idx in [0,T)
            top_global = valid_idx[top_local]  # (C, uq)

            # pad to u_q_max by repeating the first valid index
            if uq < u_q_max:
                pad = top_global[:, :1].expand(C, u_q_max - uq)
                top_global = torch.cat([top_global, pad], dim=-1)

            M_top_idx[b] = top_global

        topk_per_sample = torch.tensor(
            u_q_list,
            device=device,
            dtype=torch.long
        )  # (B,)

        return M_top_idx, topk_per_sample

    # -------------------------------------------------
    # Online clustering (unchanged core logic)
    # -------------------------------------------------
    @staticmethod
    def online_clustering(
        *,
        prototypes: torch.Tensor,   # (Class+1, K, D)
        tokens: torch.Tensor,       # (B, P, D) where P = T*C
        logits: torch.Tensor,       # (B, P, Class+1, K)
        labels: torch.Tensor,       # (B, P)
        gamma: float,
        use_sinkhorn: bool,
        use_gumbel: bool,
    ):
        B, P, D = tokens.shape
        C_total, K, _ = prototypes.shape

        tokens_flat = rearrange(tokens, "B P D -> (B P) D")
        logits_flat = rearrange(logits, "B P C K -> (B P) C K")
        labels_flat = rearrange(labels, "B P -> (B P)")

        P_old = prototypes.clone()
        P_new = prototypes.clone()

        part_assign = torch.empty_like(labels_flat)

        for c in labels_flat.unique().tolist():
            if c < 0 or c >= C_total:
                continue

            mask = labels_flat == c
            if mask.sum() == 0:
                continue

            I_c = tokens_flat[mask]          # (Nc, D)
            L_c = logits_flat[mask, c, :]    # (Nc, K)

            if use_sinkhorn:
                assign, idx = sinkhorn_knopp(L_c, use_gumbel=use_gumbel)
            else:
                idx = L_c.argmax(dim=-1)
                assign = F.one_hot(idx, num_classes=K).float()

            P_c_new = torch.mm(assign.t(), I_c)
            P_c_old = P_old[c]

            P_new[c] = momentum_update(P_c_old, P_c_new, gamma)
            part_assign[mask] = idx + c * K

        part_assign = rearrange(part_assign, "(B P) -> B P", B=B, P=P)
        return part_assign, P_new

    @staticmethod
    def build_time_weights(
            *,
            M_top_idx: torch.Tensor,  # (B, C, u_q), values in [0, T-1]
            mask: torch.Tensor,  # (B, T) bool, 1 is valid
            T: int,
            dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build normalized time weights tw from sparse indices.

        Returns:
            tw: (B, T), sum_t tw[b,t] = 1 over valid frames
        """
        B = M_top_idx.size(0)
        device = M_top_idx.device

        # accumulate counts over time
        tw_logits = torch.zeros(B, T, device=device, dtype=dtype)  # (B,T)
        idx_flat = M_top_idx.reshape(B, -1)  # (B, C*u_q)
        tw_logits.scatter_add_(
            dim=1,
            index=idx_flat,
            src=torch.ones_like(idx_flat, dtype=dtype),
        )

        # mask invalid frames before softmax
        neg_inf = torch.finfo(dtype).min
        tw_logits = tw_logits.masked_fill(~mask, neg_inf)

        tw = torch.softmax(tw_logits, dim=1)  # (B,T)
        tw = tw * mask.to(dtype=dtype)  # invalid -> 0
        tw = tw / tw.sum(dim=1, keepdim=True).clamp(min=1e-6)

        return tw

    def forward(
        self,
        hash_feat: torch.Tensor,  # (BT, C, hash_dim)
        tokens: torch.Tensor,     # (BT, C, feature_dim)
        labels: torch.Tensor,     # (B,)
        mask: torch.Tensor,       # (B, T) 0/1 or bool, 1 is valid
        *,
        B: int,
        T: int,
        update_prototypes: bool = True,
    ) -> SDPCOutput:

        BT, C, _ = hash_feat.shape
        assert BT == B * T, "BT must equal B*T"
        mask = mask.bool()

        # ---- embed hash to prototype space ----
        z = F.normalize(self.embed(hash_feat), dim=-1)  # (BT, C, D)

        # ---- reshape to (B,T,C,*) ----
        z = z.view(B, T, C, -1)
        tokens = tokens.view(B, T, C, -1)

        # video-level labels -> broadcast to (B,T,C)
        if labels.shape != (B,):
            raise ValueError(f"labels must be (B,)={(B,)}, got {labels.shape}")
        labels_tc = labels.view(B, 1, 1).expand(B, T, C).contiguous()  # (B,T,C)

        # ---- sparse measurement on time ----
        Q = z.permute(0, 2, 1, 3)  # (B, C, T, D)
        K = Q

        Q = Q * mask[:, None, :, None].to(dtype=Q.dtype)
        K = K * mask[:, None, :, None].to(dtype=K.dtype)

        M_top_idx, topk_per_sample = self._prob_QK(Q, K, mask=mask, c_factor=self.c)

        # ---- build time weights tw: (B,T) ----
        tw = self.build_time_weights(
            M_top_idx=M_top_idx,
            mask=mask,
            T=T,
            dtype=z.dtype,
        )

        # ---- build sparse mask over (C,T) ----
        mask_sparse = torch.zeros(B, C, T, dtype=torch.bool, device=z.device)
        mask_sparse.scatter_(2, M_top_idx, True)  # (B,C,T)

        # ---- assign labels: keep only sparse (t,c), others -> other_class ----
        labels_full = torch.full(
            (B, T, C),
            fill_value=self.other_class_id,
            dtype=torch.long,
            device=z.device
        )

        mask_tc = mask_sparse.permute(0, 2, 1)         # (B,T,C)
        mask_tc = mask_tc & mask[:, :, None]           # invalid frames never keep labels
        labels_full[mask_tc] = labels_tc[mask_tc]      # sparse keep original label

        # ---- flatten over (T,C): P = T*C ----
        z_all   = rearrange(z,          "B T C D -> B (T C) D")  # (B,P,D)
        tok_all = rearrange(tokens,     "B T C D -> B (T C) D")  # (B,P,D)
        lab_all = rearrange(labels_full, "B T C -> B (T C)")     # (B,P)

        # ---- prototype logits ----
        proto = F.normalize(self.prototypes, dim=-1)   # (C_total,K,D)
        z_all = F.normalize(z_all, dim=-1)             # (B,P,D)
        patch_logits = torch.einsum("bpd,ckd->bpck", z_all, proto)  # (B,P,C_total,K)

        # ---- online update ----
        if update_prototypes and self.training:
            part_assign, new_proto = self.online_clustering(
                prototypes=self.prototypes,
                tokens=tok_all,
                logits=patch_logits.detach(),
                labels=lab_all,
                gamma=self.gamma,
                use_sinkhorn=self.use_sinkhorn,
                use_gumbel=self.use_gumbel,
            )
            self.prototypes.data.copy_(new_proto)
        else:
            part_assign = patch_logits.view(B, -1, self.total_classes * self.num_prototypes).argmax(dim=-1)  # (B,P)

        # ---- reshape outputs to match SDPCOutput ----
        # patch_logits: (B,P,C_total,K) -> (B,T,C,C_total,K) -> (BT,C,C_total,K)
        patch_logits = patch_logits.view(B, T, C, self.total_classes, self.num_prototypes)
        patch_logits = patch_logits.reshape(BT, C, self.total_classes, self.num_prototypes)

        # part_assign: (B,P) -> (B,T,C) -> (BT,C)
        part_assign = part_assign.view(B, T, C).reshape(BT, C)

        return SDPCOutput(
            patch_logits=patch_logits,
            part_assignments=part_assign,
            tw=tw,
            topk_per_sample=topk_per_sample
        )


@dataclass
class KFBSDOutput:
    total: torch.Tensor          # scalar
    l_cls: torch.Tensor          # scalar
    l_ppd: torch.Tensor          # scalar
    tw: torch.Tensor             # (B, T)

class KeyFacialBehaviorSelfDistillation(nn.Module):
    """
    Key Facial Behavior Self-Distillation
    Inputs:
        x:                (BT, N, CLASS+1, K_Prototypes)   cosine similarity
        label:            (B,)  # video-level class label (0..CLASS-1)
        score:            (B,)  # video-level regression score
        part_assignments: (BT,N) or (B,T,N) or (B,T*N)  in [0, (CLASS+1)*K-1]
        tw:               (B,T)
        mask:             (B,T)  # 1 is valid
    """

    def __init__(
        self,
        *,
        n_classes: int,
        n_prototypes: int,
        init_val: float = 0.2,
        l_ppd_coef : float = 0.5,
        ppd_temp: float = 0.1,
        bg_class_weight: float = 0.1,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.C = n_classes           # clear classes (no vague)
        self.K = n_prototypes
        self.ppd_coef = l_ppd_coef
        self.ppd_temp = ppd_temp
        self.temperature = temperature

        # -------- classification aggregation (C,K) --------
        self.weights_classifier = nn.Parameter(
            torch.full((self.C, self.K), init_val, dtype=torch.float32)
        )

        self.ce = nn.CrossEntropyLoss()

        # weight for PPD CE over (CLASS+1)*K categories
        class_weights = [self.ppd_coef] * (self.C * self.K) + [float(bg_class_weight)] * self.K
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

    @staticmethod
    def apply_valid_time_weights(tw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        tw = tw * mask.to(dtype=tw.dtype)
        tw = tw / tw.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return tw

    def forward(
        self,
        *,
        x: torch.Tensor,                 # (BT, N, C_total, K) where C_total = CLASS+1
        label: torch.Tensor,             # (B,)
        score: torch.Tensor,             # (B,)
        part_assignments: torch.Tensor,  # (BT,N) or (B,T,N) or (B,T*N)
        tw: torch.Tensor,                # (B,T)
        mask: torch.Tensor,              # (B,T)
        topk_per_sample: torch.Tensor,  # (B,)
        B: int,
        T: int,
    ) -> KFBSDOutput:

        BT, N, C_total, K = x.shape
        assert BT == B * T, "BT must equal B*T"
        assert K == self.K, "K mismatch with init"
        assert C_total == self.C + 1, f"Expected C_total=CLASS+1={self.C+1}, got {C_total}"

        device = x.device
        label = label.to(device)
        score = score.to(device)

        # ensure tw respects valid frames
        tw = self.apply_valid_time_weights(tw.to(device), mask.to(device))  # (B,T)

        x = x.view(B, T, N, C_total, K)

        # ================================================================
        # (1) classification: ONLY use prototype cosine-similarity of clear CLASS (no vague)
        # ================================================================
        x_fg = x[:, :, :, : self.C, :]          # (B,T,N,CLASS,K)
        logits_tck = x_fg.mean(dim=2)           # (B,T,CLASS,K)

        # ---- classification loss ----
        sa_w = torch.softmax(self.weights_classifier, dim=-1) * K   # (CLASS,K)
        cls_tc = (logits_tck * sa_w).sum(dim=-1)                    # (B,T,CLASS)
        cls_bc = (cls_tc * tw.unsqueeze(-1)).sum(dim=1) / self.temperature  # (B,CLASS)
        l_cls = self.ce(cls_bc, label)

        # ================================================================
        # (2) PPD self-distillation: use ALL (CLASS+1) including bg
        # ================================================================
        ppd_logits = rearrange(x, "B T N C K -> B (C K) (T N)") / self.ppd_temp  # (B,(CLASS+1)*K,T*N)

        if part_assignments.dim() == 3:            # (B,T,N)
            tgt = part_assignments.reshape(B, -1)
        elif part_assignments.dim() == 2:          # (BT,N) or (B,T*N)
            if part_assignments.shape[0] == BT:    # (BT,N) -> (B, T*N)
                tgt = part_assignments.view(B, T * N)
            else:
                tgt = part_assignments
        else:
            raise ValueError(f"Unsupported part_assignments shape: {part_assignments.shape}")

        tgt = tgt.to(device).long()

        # weighted CE: vague classes down-weighted
        l_ppd = F.cross_entropy(ppd_logits, tgt, weight=self.class_weights.to(device))

        total = l_cls + l_ppd

        return KFBSDOutput(
            total=total,
            l_cls=l_cls,
            l_ppd=l_ppd,
            tw=tw.detach(),
        )

if __name__ == "__main__":
    torch.manual_seed(2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =====================================================
    # Basic settings
    # =====================================================
    B = 2          # batch size
    T = 4          # time length
    H = W = 224
    BT = B * T

    num_classes = 3
    num_prototypes = 4
    mid_dim = 64

    print(f"[Info] Running sanity check on device: {device}")

    # =====================================================
    # Dummy inputs
    # =====================================================
    video = torch.randn(B, T, 3, H, W, device=device)
    video_flat = video.view(BT, 3, H, W)

    clip_text = torch.randn(BT, 512, device=device)
    clip_image = torch.randn(BT, 768, device=device)

    mask_bt = torch.ones(B, T, device=device)
    valid_mask_bt = mask_bt.view(-1)

    labels = torch.randint(0, num_classes, (B,), device=device)
    scores = torch.randn(B, device=device)

    # =====================================================
    # 1. Backbone
    # =====================================================
    backbone = ResNet18MultiScale(pretrained=False).to(device)
    feats = backbone(video_flat)

    print("[OK] ResNet18MultiScale")
    print("  x1:", feats.x1.shape)
    print("  x2:", feats.x2.shape)
    print("  x3:", feats.x3.shape)
    print("  x4:", feats.x4.shape)

    # =====================================================
    # 2. AU-SGA
    # =====================================================
    ausga = AUSGA().to(device)
    au_feats = ausga(
        text_sem=clip_text,
        img_sem=clip_image,
        x1=feats.x1,
        x2=feats.x2,
        x3=feats.x3,
        x4=feats.x4,
    )

    print("[OK] AUSGA")

    # =====================================================
    # 3. MSR
    # =====================================================
    msr = MSR(mid_dim=mid_dim).to(device)
    msr_out = msr(
        au_feats.x1_au,
        au_feats.x2_au,
        au_feats.x3_au,
        au_feats.x4_au,
        b=B,
        t=T,
        mask=mask_bt,
    )

    print("[OK] MSR")
    print("  fmap:", msr_out.fmap.shape)
    print("  frame_emb:", msr_out.frame_emb.shape)

    # =====================================================
    # 4. PCA foreground mask
    # =====================================================
    pca_mask = PCAMask().to(device)

    # use MSR feature map as input
    fmap_btchw = msr_out.fmap.view(BT, mid_dim, *msr_out.fmap.shape[-2:])

    pca_out = pca_mask(
        x=fmap_btchw,
        valid_mask=valid_mask_bt,
    )

    print("[OK] PCAMask")
    print("  mask:", pca_out.mask_bhw.shape)

    # =====================================================
    # 5. Hash coding on foreground
    # =====================================================
    hash_out = hash_coding_foreground(
        feat=fmap_btchw,
        pool_size=8,
        topk=4,
    )

    hash_feat = hash_out.code  # (BT, C, 2*topk)
    print("[OK] Hash coding")
    print("  hash_feat:", hash_feat.shape)

    # =====================================================
    # 6. SDPC
    # =====================================================
    _, C, hash_dim = hash_feat.shape
    feature_dim = mid_dim

    sdpc = SDPC(
        num_classes=num_classes,
        num_prototypes=num_prototypes,
        hash_dim=hash_dim,
        feature_dim=feature_dim,
        c=2,
    ).to(device)

    tokens = fmap_btchw.view(BT, C, -1).mean(dim=-1)  # (BT,C,feature_dim) 简化

    sdpc_out = sdpc(
        hash_feat=hash_feat,
        tokens=tokens,
        labels=labels,
        mask=mask_bt,
        B=B,
        T=T,
    )

    print("[OK] SDPC")
    print("  patch_logits:", sdpc_out.patch_logits.shape)
    print("  part_assignments:", sdpc_out.part_assignments.shape)
    print("  tw:", sdpc_out.tw.shape)

    # =====================================================
    # 7. KFBSD loss
    # =====================================================
    kfbsd = KeyFacialBehaviorSelfDistillation(
        n_classes=num_classes,
        n_prototypes=num_prototypes,
    ).to(device)

    # fake N = channel count
    N = sdpc_out.patch_logits.shape[1]

    kfbsd_out = kfbsd(
        x=sdpc_out.patch_logits,
        label=labels,
        score=scores,
        part_assignments=sdpc_out.part_assignments,
        tw=sdpc_out.tw,
        mask=mask_bt,
        topk_per_sample=sdpc_out.topk_per_sample,  # NEW
        B=B,
        T=T,
    )

    print("[OK] KFBSD")
    print("  total loss:", float(kfbsd_out.total))

    print("\n[SUCCESS] All modules ran successfully.")

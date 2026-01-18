from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from layers import (
    KeyFacialBehaviorSelfDistillation,
    SDPC,
    PCAMask,
    MSR,
    AUSGA,
    ResNet18MultiScale,
    masked_mean,
    hash_coding_foreground,
)

# =========================================================
# Dep-MAP Main Model
# =========================================================
@dataclass
class DepMAPOutput:
    # predictions
    frame_pred: torch.Tensor  # (B, T)
    pred: torch.Tensor  # (B,) final regression output (after epoch-gate)
    loss: Optional[torch.Tensor] = None  # ()

    # debug extras
    extras: Optional[Dict[str, Any]] = None


class DepMAP(nn.Module):
    """
    Dep-MAP
    Inputs:
      video:      (B, T, 3, 224, 224)
      clip_text:  (B, T, 512)   CLIP text encoder output (NO projection)
      clip_image: (B, T, 768)   CLIP image encoder output (NO projection)
      mask:       (B, T)        1=valid, 0=pad
      score:      (B,)          regression target
      label:      (B,)          class label in [0..num_classes-1]
      epoch:      int           scalar epoch index (for gating)

    Pipeline:
      Step1  Backbone (ResNet18MultiScale) -> x1..x4
      Step2  AUSGA -> x1_au..x4_au
      Step3  MSR   -> fmap, frame_emb, frame_predict
             L_mse = MSE(masked_mean(frame_predict), score)
      Step4  SDPM (8 branches = raw1-4 + au1-4):
             - take each branch feat -> mid_dim -> resize to 7x7
             - PCAMask -> hash_coding_foreground -> SDPC -> (patch_logits, part_assign, tw)
             - only x4_au update prototypes; others update_prototypes=False
      Step5  KFBSD per-branch -> total/l_cls/l_reg/l_ppd
             if epoch>gate_epoch: loss = L_mse + lam * mean(total over 8 branches)
             else loss = L_mse

    Notes:
      - tokens for SDPC are defined as flattened 7x7 spatial tokens:
            tokens = feat_7.flatten(2)  # (BT, mid_dim, 49)
    """

    def __init__(
            self,
            *,
            num_classes: int,
            num_prototypes: int,
            mid_dim: int = 64,
            pool_size: int = 7,
            topk: int = 4,
            gate_epoch: int = 10,
            lam_kdm: float = 0.1,
            wave: str = "haar",
            backbone_pretrained: bool = True,
            pca_keep_quantile: float = 0.6,
            sdpc_c: int = 5,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.mid_dim = mid_dim

        self.pool_size = pool_size
        self.topk = topk
        self.hash_dim = 2 * topk
        self.feature_dim = 49  # 7*7 tokens
        self.gate_epoch = gate_epoch
        self.lam_kdm = lam_kdm

        # Step1
        self.backbone = ResNet18MultiScale(pretrained=backbone_pretrained)

        # Step2
        self.ausga = AUSGA()

        # Step3
        self.msr = MSR(mid_dim=mid_dim, wave=wave)

        # Step4
        self.pca_mask = PCAMask(keep_quantile=pca_keep_quantile)

        # branch projection: unify raw/au feature channels to mid_dim
        # raw/au x1..x4 channels: 64/128/256/512
        in_ch = [64, 64, 64, 64, 64, 128, 256, 512]
        self.branch_proj = nn.ModuleList([
            nn.Conv2d(c, mid_dim, kernel_size=1) for c in in_ch
        ])
        self.branch_act = nn.GELU()

        # SDPC shared across 8 branches; prototypes shared
        self.sdpc = SDPC(
            num_classes=num_classes,
            num_prototypes=num_prototypes,
            hash_dim=self.hash_dim,
            feature_dim=self.feature_dim,
            c=sdpc_c,
        )

        # Step5
        # 8 branch names (raw1-4 + au1-4)
        self.branch_names = ["raw1", "raw2", "raw3", "raw4", "au1", "au2", "au3", "au4"]

        # each branch has its own KFBSD
        self.kfbsd_branches = nn.ModuleDict({
            name: KeyFacialBehaviorSelfDistillation(
                n_classes=num_classes,
                n_prototypes=num_prototypes,
                init_val=0.2,
                l_ppd_coef=0.5,
                ppd_temp=0.1,
                bg_class_weight=0.1,
                temperature=0.1,
            )
            for name in self.branch_names
        })

        self.torch_resize = transforms.Resize((128, 128), antialias=True)

    @staticmethod
    def _resize_to_7x7(x: torch.Tensor) -> torch.Tensor:
        # x: (BT, C, H, W) -> (BT, C, 7, 7)
        if x.shape[-1] != 7 or x.shape[-2] != 7:
            x = F.adaptive_avg_pool2d(x, (7, 7))
        return x

    def _branch_feat_to_mid7(
            self,
            feat: torch.Tensor,  # (BT, Cin, H, W)
            scale_id: int,  # 0..3 for x1..x4
    ) -> torch.Tensor:
        # -> (BT, mid_dim, 7, 7)
        x = self.branch_act(self.branch_proj[scale_id](feat))
        x = self._resize_to_7x7(x)
        return x

    def _run_sdpm_branch(
            self,
            *,
            feat_mid7: torch.Tensor,  # (BT, mid_dim, 7, 7)
            valid_mask_bt: torch.Tensor,  # (BT,)
            label: torch.Tensor,  # (B,)
            mask: torch.Tensor,  # (B,T)
            B: int,
            T: int,
            update_prototypes: bool,
    ):

        # hash coding (foreground)
        hash_out = hash_coding_foreground(
            feat=feat_mid7,
            pool_size=self.pool_size,
            topk=self.topk,
        )
        hash_feat = hash_out.code  # (BT, mid_dim, 2*topk)

        # tokens: spatial flattened (BT, mid_dim, 49)
        tokens = feat_mid7.flatten(2)

        sdpc_out = self.sdpc(
            hash_feat=hash_feat,
            tokens=tokens,
            labels=label,
            mask=mask,
            B=B,
            T=T,
            update_prototypes=update_prototypes,
        )
        return sdpc_out

    @staticmethod
    def aggregate_reg_t_by_tw_topk(
            *,
            reg_t: torch.Tensor,  # (B,T)
            tw: torch.Tensor,  # (B,T) assumed already valid-normalized
            mask: torch.Tensor,  # (B,T) 0/1 or bool
            topk_per_sample: torch.Tensor,  # (B,)
            eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Select top-k frames by tw for each sample as important frames, then aggregate reg_t over them.
        We use tw-reweighted sum over the selected frames (tw renormalized inside topk).
        """
        B, T = reg_t.shape
        device = reg_t.device
        mask = mask.bool()

        # ensure tw only on valid frames
        tw = tw * mask.to(dtype=tw.dtype)
        # avoid invalid being selected: set invalid tw to -inf for topk selection
        tw_sel = tw.masked_fill(~mask, float("-inf"))

        out = torch.zeros(B, device=device, dtype=reg_t.dtype)

        for b in range(B):
            Lb = int(mask[b].sum().item())
            if Lb <= 0:
                out[b] = 0.0
                continue

            k = int(topk_per_sample[b].item())
            k = max(1, min(k, Lb))  # clamp to [1, valid_len]

            # indices of top-k frames by tw
            idx = torch.topk(tw_sel[b], k=k, dim=0, largest=True, sorted=False).indices  # (k,)

            # renormalize tw within selected frames
            w = tw[b, idx]  # (k,)
            w = w / w.sum().clamp(min=eps)

            # weighted aggregation
            out[b] = (reg_t[b, idx] * w).sum()

        return out

    def forward(
            self,
            *,
            video: torch.Tensor,  # (B,T,3,224,224)
            clip_text: torch.Tensor,  # (B,T,512)
            clip_image: torch.Tensor,  # (B,T,768)
            mask: torch.Tensor,  # (B,T) 1 valid
            score: Optional[torch.Tensor] = None,  # (B,)
            label: Optional[torch.Tensor] = None,  # (B,)
            epoch: int = 0,
            return_extras: bool = True,
    ) -> DepMAPOutput:

        device = video.device
        B, T = video.shape[0], video.shape[1]
        BT = B * T

        mask = mask.to(device)
        valid_mask_bt = mask.reshape(-1).bool()  # (BT,)

        # -------------------------------------------------
        # Step 1: Backbone
        # -------------------------------------------------
        video_flat = self.torch_resize(video.view(BT, 3, 224, 224))
        feats = self.backbone(video_flat)  # x1..x4

        # -------------------------------------------------
        # Step 2: AUSGA
        # -------------------------------------------------
        text_flat = clip_text.view(BT, 512)
        img_flat = clip_image.view(BT, 768)

        au = self.ausga(
            text_sem=text_flat,
            img_sem=img_flat,
            x1=feats.x1.clone(), x2=feats.x2.clone(), x3=feats.x3.clone(), x4=feats.x4.clone(),
        )

        # -------------------------------------------------
        # Step 3: MSR + L_mse
        # -------------------------------------------------
        msr_out = self.msr(
            feats.x1, feats.x2, feats.x3, feats.x4,
            b=B, t=T, mask=mask,
        )

        frame_pred = msr_out.frame_predict.squeeze(-1)  # (B,T)

        # masked frame aggregate: (B,)
        frame_agg = masked_mean(frame_pred, mask, dim=1)
    
        loss = None
        l_mse = None
        if score is not None:
            score = score.to(device).float()
            l_mse = F.mse_loss(frame_agg, score)

        # -------------------------------------------------
        # Step 4-5: SDPM branches + KFBSD (epoch-gated)
        # -------------------------------------------------
        use_kdm = (epoch > self.gate_epoch) and (label is not None) and (score is not None)

        tw_main = None
        branch_totals: List[torch.Tensor] = []
        branch_logs: List[Dict[str, float]] = []

        if use_kdm:
            label = label.to(device).long()
            score = score.to(device).float()

            # 8 branches: (raw1-4) + (au1-4)
            # Update prototypes only on raw4 and au4 (scale_id=3)
            branches: List[Tuple[str, torch.Tensor, int, bool]] = [
                ("au4", au.x4_au, 7, True),  # update
                ("raw1", msr_out.x1_attn, 0, False),
                ("raw2", msr_out.x2_attn, 1, False),
                ("raw3", msr_out.x3_attn, 2, False),
                ("raw4", msr_out.x4_attn, 3, False),
                ("au1", au.x1_au, 4, False),
                ("au2", au.x2_au, 5, False),
                ("au3", au.x3_au, 6, False),
            ]

            # choose tw from au4 branch by default
            for name, feat_map, sid, upd in branches:
                feat_mid7 = self._branch_feat_to_mid7(feat_map, scale_id=sid)  # (BT,mid,7,7)

                sdpc_out = self._run_sdpm_branch(
                    feat_mid7=feat_mid7,
                    valid_mask_bt=valid_mask_bt,
                    label=label,
                    mask=mask,
                    B=B,
                    T=T,
                    update_prototypes=(upd and self.training),  # only update in training
                )

                # KFBSD
                kf_out = self.kfbsd_branches[name](
                    x=sdpc_out.patch_logits,
                    label=label,
                    score=score,
                    part_assignments=sdpc_out.part_assignments,
                    tw=sdpc_out.tw,
                    mask=mask,
                    topk_per_sample=sdpc_out.topk_per_sample,
                    B=B,
                    T=T,
                )

                branch_totals.append(kf_out.total)
                branch_logs.append({
                    "l_cls": float(kf_out.l_cls.detach().cpu()),
                    "l_ppd": float(kf_out.l_ppd.detach().cpu()),
                    "total": float(kf_out.total.detach().cpu()),
                })

                if name == "au4":
                    tw_main = sdpc_out.tw  # (B,T)
                    topk_per_sample = sdpc_out.topk_per_sample

            # final loss
            kdm_loss = torch.stack(branch_totals).mean()
            loss = self.lam_kdm * kdm_loss

        else:
            loss = l_mse

        # -------------------------------------------------
        # Inference final pred (epoch-gated TW fusion)
        # -------------------------------------------------
        pred = frame_agg  # (B,)

        if (tw_main is not None) and (epoch > self.gate_epoch):
            pred = self.aggregate_reg_t_by_tw_topk(reg_t=frame_pred, tw=tw_main, mask=mask,
                                                   topk_per_sample=topk_per_sample)

            loss += F.mse_loss(pred, score)
        extras = None
        if return_extras:
            extras = {
                "frame_agg": frame_agg.detach(),
                "l_mse": None if l_mse is None else float(l_mse.detach().cpu()),
                "use_kdm": bool(use_kdm),
                "tw": None if tw_main is None else tw_main.detach(),
                "branch_logs": branch_logs,
            }

        return DepMAPOutput(
            frame_pred=frame_pred,
            pred=pred,
            loss=loss,
            extras=extras,
        )


if __name__ == "__main__":
    torch.manual_seed(2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------
    # Basic config
    # -------------------------------------------------
    B = 2
    T = 4
    H = W = 224

    num_classes = 3
    num_prototypes = 4

    print(f"[Info] Device: {device}")

    # -------------------------------------------------
    # Dummy inputs
    # -------------------------------------------------
    video = torch.randn(B, T, 3, H, W, device=device)
    clip_text = torch.randn(B, T, 512, device=device)
    clip_image = torch.randn(B, T, 768, device=device)

    mask = torch.ones(B, T, device=device)  # all frames valid
    score = torch.randn(B, device=device)  # regression target
    label = torch.randint(0, num_classes, (B,), device=device)

    # -------------------------------------------------
    # Build model
    # -------------------------------------------------
    model = DepMAP(
        num_classes=num_classes,
        num_prototypes=num_prototypes,
        mid_dim=64,
        topk=8,
        gate_epoch=10,
        lam_kdm=0.5,
        backbone_pretrained=False,  # avoid downloading weights
    ).to(device)

    # -------------------------------------------------
    # (1) Training mode, early epoch (KDM OFF)
    # -------------------------------------------------
    model.train()
    out = model(
        video=video,
        clip_text=clip_text,
        clip_image=clip_image,
        mask=mask,
        score=score,
        label=label,
        epoch=0,
    )

    print("\n[Train | epoch=0]")
    print("  frame_pred:", out.frame_pred.shape)  # (B,T)
    print("  pred:", out.pred.shape)  # (B,)
    print("  loss:", float(out.loss))

    # -------------------------------------------------
    # (2) Training mode, later epoch (KDM ON)
    # -------------------------------------------------
    out = model(
        video=video,
        clip_text=clip_text,
        clip_image=clip_image,
        mask=mask,
        score=score,
        label=label,
        epoch=20,
    )

    print("\n[Train | epoch=20]")
    print("  pred:", out.pred.shape)
    print("  loss:", float(out.loss))
    print("  use_kdm:", out.extras["use_kdm"])
    if out.extras["tw"] is not None:
        print("  tw:", out.extras["tw"].shape)  # (B,T)

    # -------------------------------------------------
    # (3) Eval mode (no loss required)
    # -------------------------------------------------
    model.eval()
    with torch.no_grad():
        out = model(
            video=video,
            clip_text=clip_text,
            clip_image=clip_image,
            mask=mask,
            epoch=20,
        )

    print("\n[Eval]")
    print("  pred:", out.pred.shape)
    print("  loss:", out.loss)

    print("\n[SUCCESS] DepMAP sanity check passed.")

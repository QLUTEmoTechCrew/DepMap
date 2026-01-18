# utils.py

from __future__ import annotations

import os
import sys
import math
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed: int = 0) -> None:
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# Simple meters / logging
# =========================================================

@dataclass
class AverageMeter:
    name: str
    sum: float = 0.0
    count: int = 0

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0].get("lr", 0.0))


# =========================================================
# Masked ops
# =========================================================

def masked_mean(
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: Union[int, Tuple[int, ...]],
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


# =========================================================
# Regression metrics (MAE / RMSE / PCC / CCC)
# =========================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    y_true, y_pred: shape (N,)
    returns: {"MAE":..., "RMSE":..., "PCC":..., "CCC":...}
    """
    y_true = np.asarray(y_true).reshape(-1).astype(np.float64)
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.float64)

    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    std_t = float(np.std(y_true))
    std_p = float(np.std(y_pred))
    if std_t < 1e-12 or std_p < 1e-12:
        pcc = 0.0
    else:
        pcc = float(np.corrcoef(y_true, y_pred)[0, 1])

    mean_t = float(np.mean(y_true))
    mean_p = float(np.mean(y_pred))
    var_t = float(np.var(y_true))
    var_p = float(np.var(y_pred))
    denom = var_t + var_p + (mean_t - mean_p) ** 2
    if denom < 1e-12:
        ccc = 0.0
    else:
        ccc = float((2.0 * pcc * std_t * std_p) / denom)

    return {"MAE": mae, "RMSE": rmse, "PCC": pcc, "CCC": ccc}


def _to_numpy_1d(x: Union[torch.Tensor, np.ndarray, List[float]]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)


# =========================================================
# Dep-MAP train / eval loops
# =========================================================

def _parse_batch_for_depmap(
    batch: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[str]]]:
    """
    Expected dataset keys:
      img_resnet: (B,T,3,224,224)
      clip_txt_enc_noproj: (B,T,512)
      clip_img_enc_noproj: (B,T,768)
      mask: (B,T) 1 valid
      score: (B,)
      label: (B,)
      id: optional, list[str] after collate
    """
    video = batch["img_resnet"].to(device, non_blocking=True)
    clip_text = batch["clip_txt_enc_noproj"].to(device, non_blocking=True)
    clip_image = batch["clip_img_enc_noproj"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)

    score = batch["score"].to(device, non_blocking=True).float()
    label = batch["label"].to(device, non_blocking=True).long()

    ids = batch.get("id", None)
    if ids is None:
        id_list = None
    elif isinstance(ids, (list, tuple)):
        id_list = list(ids)
    else:
        # fallback: tensor or other
        try:
            id_list = [str(v) for v in ids]
        except Exception:
            id_list = None

    return video, clip_text, clip_image, mask, score, label, id_list


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: Iterable[Dict[str, Any]],
    device: torch.device,
    epoch: int,
    *,
    lr_scheduler: Optional[Any] = None,
    log_interval: int = 10,
    grad_clip_norm: Optional[float] = None,
) -> Dict[str, float]:
    """
    Train one epoch for Dep-MAP regression.

    model forward signature expected:
      out = model(video=..., clip_text=..., clip_image=..., mask=..., score=..., label=..., epoch=...)
      out.loss: scalar tensor
      out.pred: (B,) tensor (optional for logging)

    returns dict with averaged losses.
    """
    model.train()

    loss_meter = AverageMeter("loss")
    mse_meter = AverageMeter("mse")  # if out.extras provides it

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), file=sys.stdout, ncols=120)

    for step, batch in pbar:
        video, clip_text, clip_image, mask, score, label, _ = _parse_batch_for_depmap(batch, device)

        optimizer.zero_grad(set_to_none=True)

        out = model(
            video=video,
            clip_text=clip_text,
            clip_image=clip_image,
            mask=mask,
            score=score,
            label=label,
            epoch=epoch,
        )

        if out.loss is None:
            raise RuntimeError("Training requires out.loss != None. Ensure score/label are passed to the model.")

        loss = out.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step}: {loss.item()}")

        loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))

        optimizer.step()
        if lr_scheduler is not None:
            # supports both per-step schedulers and per-epoch schedulers
            try:
                lr_scheduler.step()
            except TypeError:
                pass

        bs = int(video.shape[0])
        loss_meter.update(float(loss.detach().cpu()), n=bs)

        # optional: log mse from extras if you set it in DepMAPOutput.extras
        if getattr(out, "extras", None) and isinstance(out.extras, dict) and out.extras.get("l_mse", None) is not None:
            mse_meter.update(float(out.extras["l_mse"]), n=bs)

        if (step + 1) % log_interval == 0:
            lr = get_lr(optimizer)
            pbar.set_description(
                f"[train e{epoch}] loss={loss_meter.avg:.4f} mse={mse_meter.avg:.4f} lr={lr:.6f}"
            )

        # if scheduler is per-epoch, step here
        if lr_scheduler is not None:
            try:
                lr_scheduler.step()
            except TypeError:
                # already stepped per-iteration
                pass

    return {
        "loss": loss_meter.avg,
        "mse": mse_meter.avg,
        "lr": get_lr(optimizer),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: Iterable[Dict[str, Any]],
    device: torch.device,
    epoch: int,
    *,
    aggregate_by_id: bool = False,
    return_preds: bool = False,
) -> Tuple[Optional[float], Dict[str, float], Optional[Tuple[np.ndarray, np.ndarray, Optional[List[str]]]]]:
    """
    Evaluate Dep-MAP regression.

    - Computes MAE/RMSE/PCC/CCC on sample-level by default.
    - If aggregate_by_id=True and ids exist, averages predictions per id before metrics.

    returns:
      avg_loss (float or None), metrics dict, optionally (y_true, y_pred, ids)
    """
    model.eval()

    losses: List[float] = []
    y_true_list: List[np.ndarray] = []
    y_pred_list: List[np.ndarray] = []
    id_all: List[str] = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), file=sys.stdout, ncols=120)

    for step, batch in pbar:
        video, clip_text, clip_image, mask, score, label, ids = _parse_batch_for_depmap(batch, device)

        out = model(
            video=video,
            clip_text=clip_text,
            clip_image=clip_image,
            mask=mask,
            score=score,   # keep passing score/label so model can compute loss if designed so
            label=label,
            epoch=epoch,
        )

        pred = out.pred  # (B,)

        y_true_list.append(_to_numpy_1d(score))
        y_pred_list.append(_to_numpy_1d(pred))

        if ids is not None:
            id_all.extend(ids)

        if out.loss is not None:
            losses.append(float(out.loss.detach().cpu()))

        if len(losses) > 0:
            pbar.set_description(f"[valid e{epoch}] loss={float(np.mean(losses)):.4f}")

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else None

    if aggregate_by_id and len(id_all) == len(y_true):
        # group by id: average predictions and targets per id (targets usually identical across windows)
        from collections import defaultdict
        pred_map: Dict[str, List[float]] = defaultdict(list)
        true_map: Dict[str, List[float]] = defaultdict(list)
        for _id, yt, yp in zip(id_all, y_true.tolist(), y_pred.tolist()):
            pred_map[_id].append(float(yp))
            true_map[_id].append(float(yt))

        ids_unique = sorted(pred_map.keys())
        y_pred_g = np.array([np.mean(pred_map[_id]) for _id in ids_unique], dtype=np.float64)
        y_true_g = np.array([np.mean(true_map[_id]) for _id in ids_unique], dtype=np.float64)

        mets = regression_metrics(y_true_g, y_pred_g)

        if return_preds:
            return avg_loss, mets, (y_true_g, y_pred_g, ids_unique)
        return avg_loss, mets, None

    mets = regression_metrics(y_true, y_pred)
    if return_preds:
        ids_out = id_all if len(id_all) == len(y_true) else None
        return avg_loss, mets, (y_true, y_pred, ids_out)
    return avg_loss, mets, None


# =========================================================
# Optional: checkpoint I/O
# =========================================================

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {"model": model.state_dict()}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        ckpt["epoch"] = int(epoch)
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


# =========================================================
# Optional: pretty-print metrics
# =========================================================

def format_metrics(m: Dict[str, float]) -> str:
    keys = ["MAE", "RMSE", "PCC", "CCC"]
    return " ".join([f"{k}={m.get(k, float('nan')):.4f}" for k in keys])


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

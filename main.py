# main.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from dataset import My_dataset
from Dep_MAP import DepMAP

import utils  # uses train_one_epoch/evaluate/set_seed/save_checkpoint/format_metrics/build_lr_scheduler :contentReference[oaicite:1]{index=1}


# -------------------------
# Batch adapter (key rename)
# -------------------------
def adapt_batch_keys_for_utils(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert dataset keys -> utils expected keys.

    Dataset provides:
      clip_img_enc_proj / clip_txt_enc_proj
    utils expects:
      clip_img_enc_noproj / clip_txt_enc_noproj :contentReference[oaicite:2]{index=2}
    """
    if "clip_img_enc_noproj" not in batch and "clip_img_enc_proj" in batch:
        batch["clip_img_enc_noproj"] = batch["clip_img_enc_proj"]
    if "clip_txt_enc_noproj" not in batch and "clip_txt_enc_proj" in batch:
        batch["clip_txt_enc_noproj"] = batch["clip_txt_enc_proj"]
    return batch


class AdaptedLoader:
    """Wrap a DataLoader and yield adapted batches."""
    def __init__(self, loader: DataLoader):
        self.loader = loader

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield adapt_batch_keys_for_utils(batch)


# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--csv_dir", type=str, default="/public/home/data/AVEC2014/new_csv")
    p.add_argument("--npy_dir", type=str, default="/public/home/data/AVEC2014/npy", help="NPY sequence dir (T,3,H,W)")
    p.add_argument("--json_path", type=str, default="AVEC2014.json")
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--topk_au", type=int, default=5)

    # Windowing for dev/test
    p.add_argument("--test_window", type=str, default="center", choices=["first", "center", "last"])
    p.add_argument("--test_k", type=int, default=1)

    # CLIP feature level: pre/post (you want pre)
    p.add_argument("--clip_feat_level", type=str, default="pre", choices=["pre", "post"])

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Optim
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Scheduler (reuse utils.build_lr_scheduler) :contentReference[oaicite:3]{index=3}
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--warmup_factor", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-6)

    # Save/Eval
    p.add_argument("--save_dir", type=str, default="runs/depmap")
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--aggregate_by_id", action="store_true")

    # Model (keep minimal; adjust to your DepMAP signature if needed)
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--num_prototypes", type=int, default=8)
    p.add_argument("--mid_dim", type=int, default=64)

    return p.parse_args()


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    utils.set_seed(args.seed)  # :contentReference[oaicite:4]{index=4}

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.num_workers != 0:
        print("[Warn] Recommended num_workers=0 because dataset caches features in __init__.")

    # -------------------------
    # Datasets
    # -------------------------
    train_ds = My_dataset(
        mode="train",
        csv_dir=args.csv_dir,
        npy_dir=args.npy_dir,
        json_path=args.json_path,
        max_len=args.max_len,
        topk_au=args.topk_au,
        test_window=args.test_window,
        test_k=args.test_k,
        seed=args.seed,
        clip_feat_level=args.clip_feat_level,
        tqdm_enable=True,
        verbose=True,
    )

    dev_ds = My_dataset(
        mode="test",  # or "dev" 
        csv_dir=args.csv_dir,
        npy_dir=args.npy_dir,
        json_path=args.json_path,
        max_len=args.max_len,
        topk_au=args.topk_au,
        test_window=args.test_window,
        test_k=args.test_k,
        seed=args.seed + 1,
        clip_feat_level=args.clip_feat_level,
        tqdm_enable=True,
        verbose=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Wrap loaders to satisfy utils expected keys
    train_loader_adapted = AdaptedLoader(train_loader)
    dev_loader_adapted = AdaptedLoader(dev_loader)

    # -------------------------
    # Model / Optim
    # -------------------------
    model = DepMAP(
        num_classes=args.num_classes,
        num_prototypes=args.num_prototypes,
        mid_dim=args.mid_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    num_step = len(train_loader)
    scheduler = utils.create_lr_scheduler(
        optimizer=optimizer,
        num_step=num_step,
        epochs=args.epochs,
        warmup=True,
        warmup_epochs=args.warmup_epochs,
        warmup_factor=args.warmup_factor,
        end_factor=args.min_lr
    )

    # Quick sanity check (pre features often: text 512, image 768)
    try:
        s = train_ds[0]
        print("[Sanity] img_resnet:", tuple(s["img_resnet"].shape))
        print("[Sanity] clip_img_enc_proj:", tuple(s["clip_img_enc_proj"].shape), "level=", args.clip_feat_level)
        print("[Sanity] clip_txt_enc_proj:", tuple(s["clip_txt_enc_proj"].shape), "level=", args.clip_feat_level)
    except Exception as e:
        print("[Warn] sanity check failed:", repr(e))

    best_rmse: Optional[float] = None

    # -------------------------
    # Train loop (use utils.py)
    # -------------------------
    for epoch in range(1, args.epochs + 1):
        train_log = utils.train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader_adapted,
            device=device,
            epoch=epoch,
            lr_scheduler=scheduler,
            log_interval=10,
            grad_clip_norm=args.grad_clip,
        )  # :contentReference[oaicite:6]{index=6}
        print(f"[Epoch {epoch}] train: loss={train_log['loss']:.4f} mse={train_log['mse']:.4f} lr={train_log['lr']:.6f}")

        if epoch % args.eval_every == 0:
            val_loss, mets, _ = utils.evaluate(
                model=model,
                data_loader=dev_loader_adapted,
                device=device,
                epoch=epoch,
                aggregate_by_id=args.aggregate_by_id,
                return_preds=False,
            )  # :contentReference[oaicite:7]{index=7}
            msg = utils.format_metrics(mets)  # :contentReference[oaicite:8]{index=8}
            if val_loss is None:
                print(f"[Epoch {epoch}] dev : {msg}")
            else:
                print(f"[Epoch {epoch}] dev : loss={val_loss:.4f} {msg}")

            cur_rmse = float(mets.get("RMSE", 1e18))
            if best_rmse is None or cur_rmse < best_rmse:
                best_rmse = cur_rmse
                utils.save_checkpoint(
                    str(save_dir / "best.pth"),
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    extra={"best_rmse": best_rmse, "metrics": mets, "args": vars(args)},
                )  # :contentReference[oaicite:9]{index=9}
                print(f"  -> saved best.pth (RMSE={best_rmse:.4f})")

        if epoch % args.save_every == 0:
            utils.save_checkpoint(
                str(save_dir / f"epoch_{epoch}.pth"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                extra={"args": vars(args)},
            )  # :contentReference[oaicite:10]{index=10}

        utils.save_checkpoint(
            str(save_dir / "last.pth"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            extra={"args": vars(args)},
        )  # :contentReference[oaicite:11]{index=11}

    print("Done.")


if __name__ == "__main__":
    main()

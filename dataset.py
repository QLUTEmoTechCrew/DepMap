# -*- coding: utf-8 -*-
import os
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

# local CLIP implementation (OpenAI-style)
from CLIP import clip


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class WindowSpec:
    vid_idx: int
    start: int
    end: int          # exclusive
    valid_len: int    # end-start (<=T)


class My_dataset(Dataset):
    """
    __getitem__ output dict:
    {
      img_resnet:         (T,3,224,224)
      clip_img_enc_proj:  (T,D_pre_or_post)
      clip_txt_enc_proj:  (T,D_pre_or_post)
      mask:               (T,)
      score:              ()
      label:              ()
      id:                 str
    }
    """

    def __init__(
        self,
        mode: str,                      # "train"/"dev"/"test"
        csv_dir: str,
        npy_dir: str,
        json_path: str,
        max_len: int = 64,              # T
        topk_au: int = 5,
        test_window: str = "center",    # "first"/"center"/"last"
        test_k: int = 1,                # max windows per video for dev/test, <= N'
        seed: int = 2025,
        clip_name: str = "ViT-B/32",
        device: Optional[str] = None,   # None -> auto
        tqdm_enable: bool = True,
        verbose: bool = True,           # print min_len & per-video window info
        clip_feat_level: str = "post",  # "post" | "pre"
    ) -> None:
        super().__init__()
        if mode not in {"train", "dev", "test"}:
            raise ValueError(f"mode must be train/dev/test, got {mode}")
        if test_window not in {"first", "center", "last"}:
            raise ValueError(f"test_window must be first/center/last, got {test_window}")
        if max_len <= 0:
            raise ValueError("max_len must be > 0")
        if int(test_k) <= 0:
            raise ValueError("test_k must be >= 1")
        if clip_feat_level not in {"post", "pre"}:
            raise ValueError(f"clip_feat_level must be 'post' or 'pre', got {clip_feat_level}")

        self.mode = mode
        self.csv_dir = csv_dir
        self.npy_dir = npy_dir
        self.json_path = json_path
        self.T = int(max_len)
        self.topk_au = int(topk_au)
        self.test_window = test_window
        self.test_k = int(test_k)
        self.tqdm_enable = bool(tqdm_enable)
        self.verbose = bool(verbose)
        self.clip_feat_level = clip_feat_level

        self.rng = random.Random(seed)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dev = torch.device(self.device)

        # AU phrases
        self.au_phrases = self._init_au_phrases()

        # Load split
        info = load_json(json_path)
        if mode not in info:
            raise KeyError(f" predominately '{mode}' not found in json. keys={list(info.keys())}")

        self.id_list = [str(x) for x in info[mode]["IDs"]]
        self.label_list = [int(x) for x in info[mode]["labels"]]
        self.score_list = [float(x) for x in info[mode]["scores"]]
        if not (len(self.id_list) == len(self.label_list) == len(self.score_list)):
            raise ValueError("IDs/labels/scores length mismatch in json.")

        # global min_len from all CSVs -> N' = ceil(min_len / T)
        global_min_len = self._scan_global_min_len_csvdir(csv_dir)
        self.N_prime = max(1, int(math.ceil(global_min_len / self.T)))

        if self.verbose:
            print(
                f"[Global] global_min_len={global_min_len}, max_len(T)={self.T}, "
                f"N_prime=ceil(min_len/T)={self.N_prime}"
            )

        # Load per-video AU + NPY into RAM
        self._au_c_raw: List[torch.Tensor] = []
        self._au_r_raw: List[torch.Tensor] = []
        self._faces_raw: List[torch.Tensor] = []  # (Ti,3,H,W) float32 [0,1]
        self._au_names: Optional[List[str]] = None

        for sid, sc, lb in zip(self.id_list, self.score_list, self.label_list):
            key = f"{int(sc):02d}_{lb}_{sid}"
            csv_path = os.path.join(csv_dir, f"{key}.csv")
            npy_path = self._npy_path_for_sample(sc, lb, sid)

            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV not found: {csv_path}")
            if not os.path.exists(npy_path):
                raise FileNotFoundError(f"NPY not found: {npy_path}")

            au_c, au_r, au_names = self._read_au_csv(csv_path)
            if self._au_names is None:
                self._au_names = au_names
            else:
                if au_names != self._au_names:
                    raise ValueError(f"AU columns mismatch in {csv_path}")

            faces = self._load_npy_faces(npy_path)  # (Ti,3,H,W) float32 [0,1]
            if faces.shape[0] != au_c.shape[0]:
                raise ValueError(
                    f"[Frame mismatch] {key}: npy T={faces.shape[0]} vs csv T={au_c.shape[0]}"
                )

            self._au_c_raw.append(au_c)
            self._au_r_raw.append(au_r)
            self._faces_raw.append(faces)

        assert self._au_names is not None

        # Build windows (and print per-video chosen slices)
        self.windows: List[WindowSpec] = self._build_windows_with_print()

        # Init CLIP (keep fp32 to avoid LayerNorm dtype mismatch)
        self.clip_model, _ = clip.load(clip_name, device=self.device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model = self.clip_model.float()

        # Determine embedding dims depending on pre/post
        self.img_dim, self.txt_dim = self._infer_clip_dims(self.clip_model)

        # caches (CPU)
        self.cache_img_resnet: List[torch.Tensor] = []
        self.cache_clip_img: List[torch.Tensor] = []
        self.cache_clip_txt: List[torch.Tensor] = []
        self.cache_mask: List[torch.Tensor] = []
        self.cache_score: List[torch.Tensor] = []
        self.cache_label: List[torch.Tensor] = []
        self.cache_id: List[str] = []

        self._precompute_all_windows()

        # free CLIP model if desired
        del self.clip_model
        self.clip_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        

    # -------------------------
    # NPY naming: override if needed
    # -------------------------
    def _npy_path_for_sample(self, score: float, label: int, sid: str) -> str:
        name = f"{int(score):02d}_{label}_{sid}.npy"
        return os.path.join(self.npy_dir, name)

    # -------------------------
    # AU phrases
    # -------------------------
    def _init_au_phrases(self) -> Dict[str, List[str]]:
        return {
            "AU01": ["Inner brow raiser", "Eyebrows raised", "Lift eyebrows"],
            "AU02": ["Outer brow raiser", "Outer brow lift", "Outer brow arch"],
            "AU04": ["Brow lowerer", "Lower eyebrows", "Furrowed brow"],
            "AU05": ["Upper lid raiser", "Eyes widened"],
            "AU06": ["Cheek raiser", "Smile"],
            "AU07": ["Lid tightener", "Tightening of eyelids"],
            "AU09": ["Nose wrinkler", "Curl the nose"],
            "AU10": ["Upper lip raiser", "Lips apart showing teeth"],
            "AU12": ["Lip corner puller", "Grinning", "Show teeth"],
            "AU14": ["Dimpler", "Cheek dimple"],
            "AU15": ["Lip corner depressor", "Downturned corners"],
            "AU17": ["Chin raiser", "Lift the chin"],
            "AU20": ["Lip stretcher", "Nasal flaring"],
            "AU23": ["Lip tightener", "Press the lips together"],
            "AU25": ["Lips part", "Open the lips"],
            "AU26": ["Jaw drop", "Mouth stretch"],
            "AU28": ["Lip suck", "Pucker lips"],
            "AU45": ["Blink", "Eyelid closure"]
        }

    # -------------------------
    # CSV scan -> global min len
    # -------------------------
    def _scan_global_min_len_csvdir(self, csv_dir: str) -> int:
        csvs = [f for f in os.listdir(csv_dir) if f.lower().endswith(".csv")]
        if not csvs:
            raise FileNotFoundError(f"No CSV files in {csv_dir}")
        m = None
        for f in csvs:
            df = pd.read_csv(os.path.join(csv_dir, f))
            T = len(df)
            m = T if m is None else min(m, T)
        return int(m)

    # -------------------------
    # Read AU CSV (fill AU28_r=0.8 if missing)
    # -------------------------
    def _read_au_csv(self, csv_path: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        df = pd.read_csv(csv_path)
        df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

        c_cols = [c for c in df.columns if str(c).endswith("_c")]
        if not c_cols:
            raise ValueError(f"No *_c columns in {csv_path}")

        au_names = sorted([str(c)[:-2] for c in c_cols])
        c_cols_aligned = [f"{a}_c" for a in au_names]
        c_np = df[c_cols_aligned].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
        c = torch.from_numpy(c_np)  # (T, F)

        r = torch.zeros((len(df), len(au_names)), dtype=torch.float32)
        for j, a in enumerate(au_names):
            col_r = f"{a}_r"
            if col_r in df.columns:
                r_np = pd.to_numeric(df[col_r], errors="coerce").fillna(0.0).to_numpy(np.float32)
                r[:, j] = torch.from_numpy(r_np)
            else:
                r[:, j] = 0.8 if a == "AU28" else 0.0
        return c, r, au_names

    # -------------------------
    # Load NPY faces -> float32 [0,1], shape (T,3,H,W)
    # -------------------------
    def _load_npy_faces(self, npy_path: str) -> torch.Tensor:
        arr = np.load(npy_path, allow_pickle=False)
        if arr.ndim != 4:
            raise ValueError(f"NPY must be 4D (T,C,H,W) or (T,H,W,C), got {arr.shape} in {npy_path}")

        # auto-handle (T,H,W,3)
        if arr.shape[1] != 3 and arr.shape[-1] == 3:
            arr = np.transpose(arr, (0, 3, 1, 2))

        if arr.shape[1] != 3:
            raise ValueError(f"NPY channel must be 3, got {arr.shape} in {npy_path}")

        x = torch.from_numpy(arr)
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        else:
            x = x.float()
            if x.max() > 2.0:
                x = x.div_(255.0)
        return x.contiguous()

    # -------------------------
    # Windowing with prints
    # -------------------------
    def _build_windows_with_print(self) -> List[WindowSpec]:
        windows: List[WindowSpec] = []
        for vid_idx, faces in enumerate(self._faces_raw):
            Ti = int(faces.shape[0])
            sid = self.id_list[vid_idx]
            score = int(self.score_list[vid_idx])
            label = int(self.label_list[vid_idx])
            key = f"{score:02d}_{label}_{sid}"

            if self.mode in {"dev", "test"}:
                starts = self._choose_test_starts(Ti)
                starts = starts[: min(self.test_k, self.N_prime)]
            else:
                starts = self._choose_train_non_overlap_starts(Ti)
                starts = starts[: self.N_prime]

            if self.verbose:
                print(
                    f"[PerVideo] {key}: Ti={Ti}, T={self.T}, N_prime={self.N_prime}, "
                    f"chosen_k={len(starts)}, starts={starts}"
                )

            for s in starts:
                e = min(s + self.T, Ti)
                windows.append(WindowSpec(vid_idx=vid_idx, start=s, end=e, valid_len=e - s))
        return windows

    def _choose_train_non_overlap_starts(self, Ti: int) -> List[int]:
        if Ti <= self.T:
            return [0]
        max_start = Ti - self.T
        candidates = list(range(0, max_start + 1))
        self.rng.shuffle(candidates)

        picked: List[int] = []
        for s in candidates:
            ok = True
            for ps in picked:
                if not (s + self.T <= ps or ps + self.T <= s):
                    ok = False
                    break
            if ok:
                picked.append(s)
            if len(picked) >= self.N_prime:
                break
        return picked if picked else [0]

    def _choose_test_starts(self, Ti: int) -> List[int]:
        if Ti <= self.T:
            return [0]
        max_start = Ti - self.T
        k = min(self.test_k, self.N_prime)
        if k == 1:
            if self.test_window == "first":
                return [0]
            if self.test_window == "last":
                return [max_start]
            return [max_start // 2]
        return [int(round(i * max_start / (k - 1))) for i in range(k)]

    # -------------------------
    # AU -> text
    # -------------------------
    def _au_frame_to_text(self, c_1d: torch.Tensor, r_1d: torch.Tensor) -> str:
        idx = (c_1d == 1).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return "A photo of a face."
        r_sel = r_1d[idx]
        k = min(self.topk_au, int(idx.numel()))
        _, topk = torch.topk(r_sel, k=k, largest=True, sorted=True)
        chosen = idx[topk].tolist()

        phrases = []
        for j in chosen:
            au = self._au_names[j]  # type: ignore[index]
            pool = self.au_phrases.get(au, [au.lower()])
            phrases.append(self.rng.choice(pool))

        if len(phrases) == 1:
            return f"A photo of a face showing {phrases[0]}."
        return "A photo of a face showing " + ", ".join(phrases[:-1]) + ", and " + phrases[-1] + "."

    # -------------------------
    # Preprocess: one resize, two normalizes
    # -------------------------
    def _preprocess_resnet_and_clip(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (L,3,H,W) float32 in [0,1] on CPU
        returns:
          x_resnet: (L,3,224,224) float32
          x_clip:   (L,3,224,224) float32
        """
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        mean_r = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std_r  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        x_resnet = (x - mean_r) / std_r

        mean_c = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1)
        std_c  = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1)
        x_clip = (x - mean_c) / std_c

        return x_resnet, x_clip

    @staticmethod
    def _pad_repeat_last(x: torch.Tensor, T: int) -> torch.Tensor:
        if x.shape[0] == T:
            return x
        if x.shape[0] == 0:
            return torch.zeros((T,) + x.shape[1:], dtype=x.dtype)
        if x.shape[0] < T:
            pad = x[-1:].repeat(T - x.shape[0], *([1] * (x.dim() - 1)))
            return torch.cat([x, pad], dim=0)
        return x[:T]

    # -------------------------
    # CLIP: infer dims for pre/post
    # -------------------------
    @staticmethod
    def _infer_clip_dims(model: torch.nn.Module) -> Tuple[int, int]:
        """
        Returns (img_width, txt_width) for pre-proj,
        and post-proj dim can be inferred via projection matrices when needed.
        For simplicity, we treat:
          - pre text dim = model.ln_final.normalized_shape[0]
          - post text dim = model.text_projection.shape[1]
        For image:
          - if model.visual.proj exists: pre dim = proj.shape[0], post dim = proj.shape[1]
          - else: fallback post dim = 512
        """
        # text pre width
        if hasattr(model, "ln_final") and hasattr(model.ln_final, "normalized_shape"):
            txt_pre = int(model.ln_final.normalized_shape[0])
        else:
            txt_pre = 512

        # image pre width
        img_pre = None
        if hasattr(model, "visual") and hasattr(model.visual, "proj") and model.visual.proj is not None:
            img_pre = int(model.visual.proj.shape[0])
        if img_pre is None:
            # fallback (many ViT-B/32 use 512)
            img_pre = 512

        return int(img_pre), int(txt_pre)

    # -------------------------
    # CLIP encoders: pre/post
    # -------------------------
    def _encode_text_post(self, model, tokens: torch.Tensor) -> torch.Tensor:
        return model.encode_text(tokens).float().cpu()

    def _encode_image_post(self, model, image_fp16: torch.Tensor) -> torch.Tensor:
        # avoid LN dtype mismatch: cast to fp32 before entering model
        return model.encode_image(image_fp16.float()).float().cpu()

    def _encode_text_pre(self, model, tokens: torch.Tensor) -> torch.Tensor:
        """
        OpenAI CLIP-style: get pre-proj text feature (before text_projection).
        """
        # token_embedding: (N,77,width)
        x = model.token_embedding(tokens).type(model.dtype)
        x = x + model.positional_embedding.type(model.dtype)
        x = x.permute(1, 0, 2)          # NLD -> LND
        x = model.transformer(x)
        x = x.permute(1, 0, 2)          # LND -> NLD
        x = model.ln_final(x).type(torch.float32)

        # EOT position
        eot = tokens.argmax(dim=-1)
        pre = x[torch.arange(x.shape[0], device=x.device), eot]  # (N,width)
        return pre.float().cpu()

    def _encode_image_pre(self, model, image_fp16: torch.Tensor) -> torch.Tensor:
        """
        Best-effort pre-proj image feature.

        - If visual.proj exists and visual(x) returns pre, we return visual(x).
        - If visual(x) seems to return post (dim == post_dim), we approximate pre by:
            pre ≈ post @ pinv(proj)
          (works if proj is full-rank; good enough for analysis/training in practice).
        - If no proj exists, fall back to visual(x).
        """
        x = image_fp16.float()

        if not hasattr(model, "visual"):
            return model.encode_image(x).float().cpu()

        vis = model.visual
        out = vis(x)  # could be pre or post depending on implementation

        # If visual.proj exists, we can disambiguate by dimension.
        if hasattr(vis, "proj") and vis.proj is not None:
            proj = vis.proj  # (pre_dim, post_dim) in OpenAI ViT
            pre_dim = int(proj.shape[0])
            post_dim = int(proj.shape[1])

            if out.shape[-1] == pre_dim:
                return out.float().cpu()

            if out.shape[-1] == post_dim:
                pinv = torch.linalg.pinv(proj.float())  # (post_dim, pre_dim)
                pre = out.float() @ pinv                # (N, pre_dim)
                return pre.cpu()

            return out.float().cpu()

        return out.float().cpu()

    # -------------------------
    # Precompute all windows
    # -------------------------
    def _precompute_all_windows(self) -> None:
        if self.clip_model is None:
            raise RuntimeError("CLIP model is not initialized.")

        model = self.clip_model
        dev = self.dev

        it = enumerate(self.windows)
        if self.tqdm_enable:
            it = tqdm(
                it,
                total=len(self.windows),
                desc=f"[{self.mode}] Precompute (NPY->ResNet+CLIP, feat={self.clip_feat_level})",
                ncols=120,
            )

        with torch.no_grad():
            for wi, w in it:
                sid = self.id_list[w.vid_idx]
                score = float(self.score_list[w.vid_idx])
                raw_label = int(self.label_list[w.vid_idx])
                label = 2 if raw_label >= 2 else (0 if raw_label < 0 else raw_label)  # clamp to {0,1,2}


                if self.tqdm_enable and hasattr(it, "set_postfix"):
                    it.set_postfix({"vid": sid, "win": f"{w.start}:{w.end}", "len": w.valid_len})

                faces_full = self._faces_raw[w.vid_idx]   # (Ti,3,H,W) CPU float32
                au_c_full = self._au_c_raw[w.vid_idx]     # (Ti,F)
                au_r_full = self._au_r_raw[w.vid_idx]     # (Ti,F)

                faces = faces_full[w.start:w.end]         # (L,3,H,W)
                c_win = au_c_full[w.start:w.end]
                r_win = au_r_full[w.start:w.end]

                # mask
                mask = torch.zeros((self.T,), dtype=torch.float32)
                mask[: w.valid_len] = 1.0

                # preprocess
                img_resnet_valid, clip_img_valid_cpu = self._preprocess_resnet_and_clip(faces)  # (L,3,224,224)
                img_resnet = self._pad_repeat_last(img_resnet_valid, self.T)                    # (T,3,224,224)

                # texts + tokens
                texts = [self._au_frame_to_text(c_win[i], r_win[i]) for i in range(w.valid_len)]
                tok = clip.tokenize(texts).to(dev, non_blocking=True).long()                    # (L,77)

                # image to GPU half first (as requested)
                clip_img_valid = clip_img_valid_cpu.to(dev, non_blocking=True).half()          # fp16

                # encode CLIP features (pre or post)
                if self.clip_feat_level == "post":
                    img_feat = self._encode_image_post(model, clip_img_valid)  # (L,D_post)
                    txt_feat = self._encode_text_post(model, tok)              # (L,D_post)
                else:
                    img_feat = self._encode_image_pre(model, clip_img_valid)   # (L,D_pre)
                    txt_feat = self._encode_text_pre(model, tok)               # (L,D_pre)

                # pad features to T (repeat last)
                D_img = int(img_feat.shape[1])
                D_txt = int(txt_feat.shape[1])

                clip_img = torch.zeros((self.T, D_img), dtype=torch.float32)
                clip_txt = torch.zeros((self.T, D_txt), dtype=torch.float32)
                clip_img[: w.valid_len] = img_feat
                clip_txt[: w.valid_len] = txt_feat

                if w.valid_len < self.T:
                    clip_img[w.valid_len:] = clip_img[w.valid_len - 1:w.valid_len].repeat(self.T - w.valid_len, 1)
                    clip_txt[w.valid_len:] = clip_txt[w.valid_len - 1:w.valid_len].repeat(self.T - w.valid_len, 1)

                # cache
                self.cache_img_resnet.append(img_resnet.contiguous())
                self.cache_clip_img.append(clip_img.contiguous())
                self.cache_clip_txt.append(clip_txt.contiguous())
                self.cache_mask.append(mask.contiguous())
                self.cache_score.append(torch.tensor(score, dtype=torch.float32))
                self.cache_label.append(torch.tensor(label, dtype=torch.long))
                self.cache_id.append(sid)

    # -------------------------
    # Dataset protocol
    # -------------------------
    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "img_resnet": self.cache_img_resnet[idx],
            "clip_img_enc_proj": self.cache_clip_img[idx],
            "clip_txt_enc_proj": self.cache_clip_txt[idx],
            "mask": self.cache_mask[idx],
            "score": self.cache_score[idx],
            "label": self.cache_label[idx],
            "id": self.cache_id[idx],
        }


if __name__ == "__main__":
    ds = My_dataset(
        mode="train",
        csv_dir="/public/home/acw92jjscn/wh/data/AVEC2014/new_csv",
        npy_dir="/public/home/acw92jjscn/wh/data/AVEC2014/npy",
        json_path="AVEC2014.json",
        max_len=64,
        topk_au=5,
        test_window="center",
        test_k=1,
        seed=2025,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tqdm_enable=True,
        verbose=True,
        clip_feat_level="post",  # or "pre"
    )

    print("len(ds) =", len(ds))
    x = ds[0]
    for k, v in x.items():
        if torch.is_tensor(v):
            print(k, tuple(v.shape), v.dtype, v.device)
        else:
            print(k, v)

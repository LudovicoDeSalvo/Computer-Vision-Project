#!/usr/bin/env python3
"""eval.py — Evaluation / inference utility for the Parallel Decoding LPR network.

Usage examples
--------------
$ python -m pdlpr.eval \
        --data_root /data/CCPD2019 \
        --weights runs/PDLPR/best.ckpt \
        --batch 256 --device cuda:0

Outputs an evaluation table with **character‑level accuracy**, **plate‑level
accuracy** and **mean decoding time** (FPS).

If you already have YOLO detections saved as crops or their bounding box
coordinates, point ``--crop_dir`` to the folder of 48×144 crops and run the same
command — the script does not require raw CCPD file‑name parsing in that case.

The implementation mirrors the training utilities so that results are
reproducible with a single checkpoint and the original dataset.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image


# -----------------------------------------------------------------------------
#  Character set (must match training)
# -----------------------------------------------------------------------------

CHINESE = (
    "皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤",
    "桂","琼","川","贵","云","藏","陕","甘","青","宁","新","警","学","O",
)
LETTERS = tuple("ABCDEFGHJKLMNPQRSTUVWXYZO") 
DIGITS = tuple("0123456789")

# ----- CCPD index decoding helper (added automatically) ----------------
_PROVINCES = [
    "皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","京","闽","赣","鲁",
    "豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新",
    "警","学","O",
]
_ALPHABETS = list("ABCDEFGHJKLMNPQRSTUVWXYZO")   # include O
_ADS       = _ALPHABETS + list("0123456789") + ["O"]

def _decode_plate(indices: str) -> str | None:
    """Convert CCPD index string like '0_0_22_27_27_33_16' → '皖A04025'."""
    try:
        arr = [int(x) for x in indices.split("_")]
        if len(arr) != 7:
            return None
        pro = _PROVINCES[arr[0]]
        alp = _ALPHABETS[arr[1]]
        tail = "".join(_ADS[i] for i in arr[2:])
        # remove padding 'O'
        tail = tail.replace("O", "")
        return pro + alp + tail
    except (ValueError, IndexError):
        return None
# ----------------------------------------------------------------------

SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"

VOCAB = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + list(CHINESE + LETTERS + DIGITS)
VOCAB_DICT = {ch: i for i, ch in enumerate(VOCAB)}

PAD_ID = VOCAB_DICT[PAD_TOKEN]
SOS_ID = VOCAB_DICT[SOS_TOKEN]
EOS_ID = VOCAB_DICT[EOS_TOKEN]


# -----------------------------------------------------------------------------
#  Dataset
# -----------------------------------------------------------------------------



class CCPDPlateDataset(Dataset):
    def __init__(self, root: str | pathlib.Path, height: int = 48, width: int = 144):
        self.root = pathlib.Path(root)
        self.height, self.width = height, width

        self.imgs = sorted(
            [p for p in self.root.rglob("*.jpg") if p.is_file() or p.suffix.lower() in {".jpg", ".png"}],
            key=lambda p: p.name,
        )
        if not self.imgs:
            raise FileNotFoundError(f"No images found under {self.root!s}")

        # Don't include ToTensor since we already return a tensor from read_image
        self.tf = T.Compose([
            T.Resize((self.height, self.width)),
            T.ToTensor(),  # convert PIL or array to [0,1] tensor
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        p = self.imgs[idx]
        # load and preprocess
        pil = Image.open(p).convert("RGB")
        img = self.tf(pil)   # Resize → ToTensor → Normalize

        # extract and encode the plate string
        label = self._extract_plate_from_filename(p)
        encoded = self.encode(label)

        return img, encoded, p.name


    
    def _extract_plate_from_filename(self, p: pathlib.Path) -> str:
        """Extract plate string from filename, supporting CCPD and plate-named crops."""
        name = p.stem
        parts = name.split("-")
        if len(parts) >= 5 and "_" in parts[4]:
            # CCPD encoded format (e.g., -xy-xy-xy-ABC123_*)
            plate = _decode_plate(parts[4])
            if plate:
                return plate
        if "-" in name:
            try:
                return name.split("-")[1].split("_")[0]
            except (IndexError, ValueError):
                pass
        return name
    
    @staticmethod
    def encode(text: str) -> List[int]:
        """Convert a plate string into a list of vocab indices."""
        return [VOCAB_DICT[c] for c in text if c in VOCAB_DICT]

    @staticmethod
    def decode(ids: List[int]) -> str:
        """Raw decoding (for debug) — does not skip special tokens."""
        return "".join(VOCAB[i] for i in ids if i < len(VOCAB))

    @staticmethod
    def decode_filtered(ids: List[int]) -> str:
        """Filtered decoding — used for accuracy computation."""
        chars = []
        for i in ids:
            if i == EOS_ID:
                break
            if i in (PAD_ID, SOS_ID):
                continue
            chars.append(VOCAB[i])
        return "".join(chars)


# -----------------------------------------------------------------------------
#  Collate / batching
# -----------------------------------------------------------------------------

def collate_fn(batch):
    imgs, labels, names = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    max_len = max(len(l) for l in labels)
    padded = torch.full((len(labels), max_len), PAD_ID, dtype=torch.long)
    for i, l in enumerate(labels):
        padded[i, : len(l)] = torch.tensor(l, dtype=torch.long)
    return imgs, padded, names



# -----------------------------------------------------------------------------
#  Metric helpers
# -----------------------------------------------------------------------------

# your evaluate function

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str | torch.device):
    model.eval()
    t_start = time.time()
    plate_correct = 0
    char_correct = 0
    char_total = 0

    for imgs, labels, names in tqdm(loader, desc="Eval", unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        pred_ids = model.inference(
            imgs,
            sos_id=SOS_ID,
            eos_id=EOS_ID,
            device=device
        )

        # debug sanity check once
        if plate_correct == 0 and char_total == 0:
            print("\n🔍 Sanity check:")
            for name, gt_ids, pred in list(zip(names, labels, pred_ids))[:5]:
                raw_gt   = CCPDPlateDataset.decode(gt_ids)
                raw_pred = CCPDPlateDataset.decode(pred.tolist())
                print(f"{name:<25} | GT: {raw_gt} | PRED: {raw_pred}")
            print()

        # compute accuracy
        for gt_ids, pred in zip(labels, pred_ids):
            gt_text   = CCPDPlateDataset.decode_filtered(gt_ids)
            pred_text = CCPDPlateDataset.decode_filtered(pred.tolist())
            plate_correct += int(pred_text == gt_text)
            char_correct  += sum(p == g for p, g in zip(pred_text, gt_text))
            char_total    += max(len(pred_text), len(gt_text))

    duration = time.time() - t_start
    fps = len(loader.dataset) / duration
    plate_acc = plate_correct / len(loader.dataset) * 100
    char_acc  = char_correct / char_total * 100
    return plate_acc, char_acc, fps



# -----------------------------------------------------------------------------
#  Main entry
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="PDLPR evaluation script")
    parser.add_argument("--data_root", type=str, required=True, help="Path to CCPD images or plate crops folder")
    parser.add_argument("--weights", type=str, required=True, help="Path to .ckpt/.pt weights file")
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()



def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load dataset
    ds = CCPDPlateDataset(args.data_root)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # Load model
    from pdlpr.model import PDLPR

    model = PDLPR(num_classes=len(VOCAB))
    ckpt = torch.load(args.weights, map_location="cpu")

    # Extract the actual state dict (your train.py writes it under "model")
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "model" in ckpt:
        state = ckpt["model"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt  # raw checkpoint

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"✔️  Loaded weights from {args.weights}")
    print(f"   · missing keys   : {len(missing)}")
    print(f"   · unexpected keys: {len(unexpected)}")
    if missing:
        print("   First 5 missing →", missing[:5])

    model.to(device)

    plate_acc, char_acc, fps = evaluate(model, dl, device)

    print("\nEvaluation results")
    print("------------------")
    print(f"Plate accuracy : {plate_acc:6.2f} %")
    print(f"Char accuracy  : {char_acc:6.2f} %")
    print(f"Throughput     : {fps:6.1f} FPS (plates/s)\n")



if __name__ == "__main__":
    main()

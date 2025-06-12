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

from pdlpr.train import Tokenizer 


# -----------------------------------------------------------------------------
#  Character set (must match training)
# -----------------------------------------------------------------------------

_PROVINCES = [
    "皖","沪","津","渝","冀","晋","辽","吉","黑","苏","浙","京","闽","赣","鲁",
    "豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新",
]
_ALPHABETS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
_ADS = _ALPHABETS + list("0123456789") + ["O"]

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



# -----------------------------------------------------------------------------
#  Dataset
# -----------------------------------------------------------------------------



class CCPDPlateDataset(Dataset):
    def __init__(self, root: str | pathlib.Path, tokenizer: Tokenizer, height: int = 48, width: int = 144):
        self.root = pathlib.Path(root)
        self.height, self.width = height, width
        self.tokenizer = tokenizer

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
        pil = Image.open(p).convert("RGB")
        img = self.tf(pil)

        # extract and encode the plate string
        label = self._extract_plate_from_filename(p)
        encoded = self.tokenizer.encode(label)[1:-1]

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


# -----------------------------------------------------------------------------
#  Collate / batching
# -----------------------------------------------------------------------------

def collate_fn(batch, pad_id: int):
    imgs, labels, names = zip(*batch)
    imgs = torch.stack(imgs, dim=0)

    # Convert labels (which are lists of ints) to tensors before padding
    labels = [torch.tensor(l, dtype=torch.long) for l in labels]

    max_len = max(len(l) for l in labels)
    padded = torch.full((len(labels), max_len), pad_id, dtype=torch.long)
    for i, l in enumerate(labels):
        padded[i, : len(l)] = l
    return imgs, padded, names



# -----------------------------------------------------------------------------
#  Metric helpers
# -----------------------------------------------------------------------------

# your evaluate function

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, tokenizer: Tokenizer, device: str | torch.device):
    model.eval()
    t_start = time.time()
    plate_correct = 0
    char_correct = 0
    char_total = 0

    for imgs, labels, names in tqdm(loader, desc="Eval", unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        # Use the tokenizer for special IDs
        pred_ids = model.inference(
            imgs,
            sos_id=tokenizer.sos_id,
            eos_id=tokenizer.eos_id,
            device=device
        )

        # compute accuracy
        for gt_ids, pred in zip(labels, pred_ids):
            # Use the tokenizer to decode
            gt_text   = tokenizer.decode(gt_ids.tolist())
            pred_text = tokenizer.decode(pred.tolist())
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

    # Load model
    from pdlpr.model import PDLPR
    ckpt = torch.load(args.weights, map_location="cpu")

    if "tokenizer" not in ckpt:
        raise ValueError("The provided checkpoint is old and does not contain a tokenizer.")
    
    tokenizer = Tokenizer()
    tokenizer.__dict__.update(ckpt["tokenizer"])
    print(f"✔️  Loaded tokenizer (vocab size: {tokenizer.vocab_size})")

    # Load dataset, passing the tokenizer
    ds = CCPDPlateDataset(args.data_root, tokenizer=tokenizer)
    
    # Use a lambda to pass the correct pad_id to the collate function
    dl = DataLoader(
        ds, 
        batch_size=args.batch, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_id) # ❗ Use lambda here
    )
    
    model = PDLPR(num_classes=tokenizer.vocab_size) # 🔄 Use loaded vocab size

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

    plate_acc, char_acc, fps = evaluate(model, dl, tokenizer, device) # 🔄 Pass it here

    print("\nEvaluation results")
    print("------------------")
    print(f"Plate accuracy : {plate_acc:6.2f} %")
    print(f"Char accuracy  : {char_acc:6.2f} %")
    print(f"Throughput     : {fps:6.1f} FPS (plates/s)\n")



if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import itertools
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.nn import CTCLoss
from tqdm import tqdm

from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split

# ------------ CCPD index  ----------------

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


# ----------------------------------------
# Tokenizer
# ----------------------------------------

class Tokenizer:
    def __init__(self):
        
        provinces = (
            "皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学O"
        )

        if len(provinces) != 34:
            raise ValueError("Province list length mismatch (expected 34)")

        letters = "ABCDEFGHJKLMNPQRSTUVWXYZO"
        digits = "0123456789"

        special = ["<PAD>", "<SOS>", "<EOS>"]
        self.idx2char: List[str] = special + list(provinces) + list(letters) + list(digits)
        self.char2idx = {c: i for i, c in enumerate(self.idx2char)}

        self.pad_id = self.char2idx["<PAD>"]
        self.sos_id = self.char2idx["<SOS>"]
        self.eos_id = self.char2idx["<EOS>"]

    @property
    def vocab_size(self) -> int:
        return len(self.idx2char)

    def encode(self, plate: str) -> List[int]:
        """Encode **raw plate string** (e.g. "京A12345") to integer tokens with
        explicit <SOS>/<EOS>.
        """
        seq = [self.sos_id]
        for ch in plate:
            try:
                seq.append(self.char2idx[ch])
            except KeyError as exc:
                raise KeyError(f"Unexpected character '{ch}' in plate '{plate}'") from exc
        seq.append(self.eos_id)
        return seq

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        out = []
        for i in ids:
            if strip_special and i in {self.pad_id, self.sos_id, self.eos_id}:
                continue
            out.append(self.idx2char[i])
        return "".join(out)

# ----------------------------------------
# CCPD2019 Dataset
# ----------------------------------------

class CCPDDataset(Dataset):

    IMG_EXTS = {".jpg", ".png", ".jpeg", ".bmp"}

    def __init__(self, root: str | Path, tokenizer: Tokenizer, transform: transforms.Compose | None = None):
        self.root = Path(root)
        self.transform = transform
        self.tokenizer = tokenizer

        self.samples: List[Tuple[Path, List[int]]] = []
        for p in self.root.rglob("*"):
            if p.suffix.lower() not in self.IMG_EXTS:
                continue
            label = self._parse_label(p.name)
            if label is None:
                continue
            token_ids = self.tokenizer.encode(label)
            self.samples.append((p, token_ids))

        if not self.samples:
            raise RuntimeError(f"No valid CCPD images found under {self.root}")


    def _parse_label(self, fname: str) -> str | None:
        """Parse plate from CCPD filename (handles both index‑encoded and Unicode)"""
        base = os.path.splitext(fname)[0]
        parts = base.split("-")
        if len(parts) >= 5 and "_" in parts[4]:
            plate = _decode_plate(parts[4])
            if plate:
                return plate
        # fallback to old unicode‑in‑filename format
        try:
            plate_part = fname.split("-")[1]
            return plate_part.split("_")[0]
        except IndexError:
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, token_ids = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, torch.tensor(token_ids, dtype=torch.long)

# ----------------------------------------
# Collate function 
# ----------------------------------------

def collate_fn(batch, pad_id: int):
    imgs, seqs = zip(*batch)

    # images -> stacked tensor (B,C,H,W)
    imgs = torch.stack(imgs, dim=0)

    # pad sequences to max length in batch
    max_len = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        end = len(s)
        padded[i, :end] = s
    return imgs, padded

# ----------------------------------------
#  Loss Function
# ----------------------------------------

@torch.no_grad()
def eval_ctc_decode(model: nn.Module,
                    loader: DataLoader,
                    tokenizer: Tokenizer,
                    device: torch.device) -> tuple[float, float]:
    """Evaluates model performance using CTC greedy decoding."""
    model.eval()
    plate_ok = char_ok = char_total = 0
    blank_id = tokenizer.pad_id

    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        
        # Get logits in (T, B, V) format from the CTC head
        logits_t_b_v = model.forward_ctc(imgs) #
        pred_indices = logits_t_b_v.argmax(dim=-1).T  # Transpose to (B, T)

        # Decode each prediction in the batch
        for i in range(pred_indices.size(0)):
            # 1. Greedy decode and remove repeats
            pred_collapsed = [k for k, _ in itertools.groupby(pred_indices[i].tolist())]
            
            # 2. Remove blank tokens
            pred_decoded_ids = [p for p in pred_collapsed if p != blank_id]
            
            # 3. Convert to string
            predicted_plate = tokenizer.decode(pred_decoded_ids, strip_special=False)
            
            # Get ground truth text (strip SOS/EOS)
            gt_ids = targets[i][1:-1]
            ground_truth_plate = tokenizer.decode(gt_ids.tolist())

            # Compare and calculate accuracy
            if predicted_plate == ground_truth_plate:
                plate_ok += 1
            char_ok += sum(a == b for a, b in zip(predicted_plate, ground_truth_plate))
            char_total += max(len(predicted_plate), len(ground_truth_plate))

    plate_acc = plate_ok / len(loader.dataset) if len(loader.dataset) > 0 else 0
    char_acc = char_ok / char_total if char_total > 0 else 0
    model.train()
    return plate_acc, char_acc

# ----------------------------------------
# Scheduler helper
# ----------------------------------------

def build_scheduler(optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int):
    """Linear warm‑up followed by cosine decay"""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ----------------------------------------
# Main training routine
# ----------------------------------------

def train(args):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # 1. Tokenizer
    tokenizer = Tokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    # 2. Data transforms
    train_tfms = transforms.Compose([
        transforms.Resize((48, 144)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomAffine(degrees=4, translate=(0.02, 0.06), scale=(0.9, 1.1), shear=4, fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((48, 144)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    # 3. Dataset & loaders
    if args.train_root and args.val_root:
        train_ds = CCPDDataset(args.train_root, tokenizer, transform=train_tfms)
        val_ds = CCPDDataset(args.val_root, tokenizer, transform=val_tfms)
    else:
        full_ds = CCPDDataset(args.data_root, tokenizer, transform=train_tfms)
        val_size = int(len(full_ds) * args.val_split)
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size])
        val_ds.dataset.transform = val_tfms  # type: ignore[attr-defined]

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
    )

    # 4. Model initialization
    from pdlpr.model import PDLPR
    model = PDLPR(num_classes=tokenizer.vocab_size)
    model.to(device)

    # 5. Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    # 6. Criterion: CTC Loss
    criterion = nn.CTCLoss(blank=tokenizer.pad_id, zero_infinity=True)

    # 7. Resume checkpoint if any
    start_epoch = 0
    best_acc = 0.0
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"✔️  Resumed from {args.resume} (epoch {start_epoch})")

    # 8. Training loop
    patience, patience_counter = args.early_stop_patience, 0
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")

        for step, (imgs, targets) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits_t_b_v = model.forward_ctc(imgs)
            targets_ctc = targets[:, 1:-1]

            B = imgs.size(0)
            T = logits_t_b_v.size(0)
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
            target_lengths = (targets_ctc != tokenizer.pad_id).sum(dim=1)

            with autocast(enabled=scaler.is_enabled()):
                log_probs = F.log_softmax(logits_t_b_v, dim=-1) 
                loss = criterion(log_probs, targets_ctc, input_lengths, target_lengths) 

            scaler.scale(loss).backward()
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix_str(f"loss={running_loss / (step + 1):.8f}")


        avg_loss = running_loss / len(train_loader)
        plate_acc, char_acc = eval_ctc_decode(model, val_loader, tokenizer, device)
        print(f"Epoch {epoch+1:>2} ┃ Loss: {avg_loss:.8f} ┃ Plate Acc: {plate_acc*100:6.4f}% ┃ Char Acc: {char_acc*100:6.4f}%")


        #Early Stopping
        is_best = plate_acc > best_acc
        if is_best:
            best_acc = plate_acc
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⏳ No improvement. Patience: {patience_counter}/{patience}.")
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered after {patience} epochs without improvement.")
                break

        # Checkpointing
        if (epoch + 1) % args.save_every == 0 or is_best:
            ckpt_name = "best.pt" if is_best else f"epoch{epoch+1}.pt"
            ckpt_path = Path(args.out_dir) / ckpt_name
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_acc": best_acc,
                "tokenizer": tokenizer.__dict__,
            }, ckpt_path)
            if is_best:
                print(f"✅ Saved best model: {ckpt_path.resolve().relative_to(Path.cwd().resolve())}")
            else:
                print(f"💾 Saved checkpoint: {ckpt_path.resolve().relative_to(Path.cwd().resolve())}")

    print("🏁 Training finished!")
    print(f"Best validation accuracy: {best_acc*100:.2f}%")

# ----------------------------------------
# CLI
# ----------------------------------------

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train PDLPR network on CCPD2019")
    parser.add_argument("--data_root", type=str, required=True, help="Path to CCPD dataset root")
    parser.add_argument("--out_dir", type=str, default="pdlpr", help="Directory to store checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Total number of epochs")
    parser.add_argument("--warmup_epochs", type=float, default=1.0, help="Warm‑up duration in epochs")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping (0 = disabled)")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of training set used for validation")
    parser.add_argument("--train_root", type=str, default="", help="(optional) path to training images folder")
    parser.add_argument("--val_root",   type=str, default="", help="(optional) path to validation images folder")
    parser.add_argument("--no_amp", dest="amp", action="store_false", help="Disable mixed-precision training")
    parser.set_defaults(amp=True)
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--save_every", type=int, default=20, help="Checkpoint every N epochs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--early_stop_patience", type=int, default=5, help="Number of epochs with no improvement to wait before stopping")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)
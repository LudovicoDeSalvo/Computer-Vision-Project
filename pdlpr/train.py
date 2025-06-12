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
import torchvision.transforms as T
from PIL import Image

from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split

# ----- CCPD index decoding helper (added automatically) ----------------
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
# Tokeniser / Vocabulary
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

    def __init__(self, root: str | Path, tokenizer: Tokenizer, transform: T.Compose | None = None):
        self.root = Path(root)
        self.transform = transform
        self.tokenizer = tokenizer

        self.samples: List[Tuple[Path, List[int]]] = []
        for p in self.root.rglob("*"):
            if p.suffix.lower() not in self.IMG_EXTS:
                continue
            label = self._parse_label(p.name)
            if label is None:
                continue  # ignore weird files
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
            img = T.ToTensor()(img)
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
#  Training utils
# ----------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, tokenizer: Tokenizer, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # prepare shifted inputs/labels (teacher forcing)
        tgt_in = targets[:, :-1]
        tgt_out = targets[:, 1:]

        logits = model(imgs, tgt_in)  # (B,L,V)
        pred = logits.argmax(dim=-1)
        # compare full sequence equality (excluding PAD)
        for p, t in zip(pred, tgt_out):
            # strip after EOS in target
            eos_positions = (t == tokenizer.eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions):
                t_len = eos_positions[0].item() + 1  # include EOS
            else:
                t_len = len(t)
            if torch.equal(p[:t_len], t[:t_len]):
                correct += 1
            total += 1
    model.train()
    return correct / total if total else 0.0

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
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # 1. Tokenizer
    tokenizer = Tokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    # 2. Data pipeline
    train_tfms = T.Compose([
        T.Resize((48, 144)),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomAffine(degrees=4, translate=(0.02, 0.06),
                       scale=(0.9, 1.1), shear=4, fill=0),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    val_tfms = T.Compose([
        T.Resize((48, 144)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    # If user passed explicit train/val roots, use those folders directly.
    if args.train_root and args.val_root:
        print(f"▶️  Using explicit train set: {args.train_root}")
        print(f"▶️  Using explicit val   set: {args.val_root}")
        train_ds = CCPDDataset(args.train_root, tokenizer, transform=train_tfms)
        val_ds   = CCPDDataset(args.val_root,   tokenizer, transform=val_tfms)
    else:
        # old behavior: random split from single data_root
        full_ds = CCPDDataset(args.data_root, tokenizer, transform=train_tfms)
        val_size = int(len(full_ds) * args.val_split)
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size])
        # override val split transforms
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

    # 3. Model
    from pdlpr.model import PDLPR

    model = PDLPR(num_classes=tokenizer.vocab_size)
    model.to(device)

    # 4. Optimiser & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    # 5. Criterion
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # 6. Optionally resume
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

    # Initialize early stopping
    patience = args.early_stop_patience
    patience_counter = 0

    # 7. Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        for step, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # teacher forcing: shift
            tgt_in = targets[:, :-1]
            tgt_out = targets[:, 1:]

            with autocast(enabled=scaler.is_enabled()):
                logits = model(imgs, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            scaler.scale(loss).backward()
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            if step % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{args.epochs} ┃ Step {step:05d}/{len(train_loader)} ┃ Loss {loss.item():.4f} ┃ LR {lr:.6e}")

        avg_loss = epoch_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, tokenizer, device)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}, val_acc={val_acc*100:.2f}%")

        # Early stopping check
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered after {patience} epochs without improvement.")
                break

        # Checkpoint
        if (epoch + 1) % args.save_every == 0 or is_best:
            ckpt_path = Path(args.out_dir) / ("best.pt" if is_best else f"epoch{epoch+1}.pt")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_acc": best_acc,
                    "tokenizer": tokenizer.__dict__,
                },
                ckpt_path,
            )
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
    parser.add_argument("--epochs", type=int, default=50, help="Total number of epochs")
    parser.add_argument("--warmup_epochs", type=float, default=1.0, help="Warm‑up duration in epochs")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping (0 = disabled)")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of training set used for validation")
    parser.add_argument("--train_root", type=str, default="", help="(optional) path to training images folder")
    parser.add_argument("--val_root",   type=str, default="", help="(optional) path to validation images folder")
    parser.add_argument("--no_amp", dest="amp", action="store_false", help="Disable mixed-precision training")
    parser.set_defaults(amp=True)
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--save_every", type=int, default=5, help="Checkpoint every N epochs")
    parser.add_argument("--log_every", type=int, default=50, help="Step interval for logging")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--early_stop_patience", type=int, default=5, help="Number of epochs with no improvement to wait before stopping")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)
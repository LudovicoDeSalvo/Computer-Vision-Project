# evaluate.py – SOLO inference CTC (nessun training)
# --------------------------------------------------------------------------------
"""Inferisce un modello OCR‑CTC già addestrato su un sotto‑dataset di
crop CCPD e calcola la similarità Levenshtein pred↔GT.

Lo script è pensato per essere richiamato da *main.py* con:

    python baseline/evaluate.py \
        --data_root  <cartella_crop> \
        --weights    baseline/ocr_model.pth \
        --batch      64
"""

from __future__ import annotations
import argparse, pathlib, re, torch, pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from difflib import SequenceMatcher

from model import OCRModel  # stessa architettura usata al training

# ─────────────────── TOKENIZER CCPD (identico a train.py) ───────────────────────
_PROVINCES = [
    "皖","沪","津","渝","冀","晋","辽","吉","黑","苏","浙","京","闽","赣","鲁",
    "豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新",
]
_ALPHABETS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
_ADS       = _ALPHABETS + list("0123456789") + ["O"]

CTC_CHARS  = _PROVINCES + _ALPHABETS + list("0123456789")      # senza padding (blank=0)
IDX2CHAR   = {i+1: c for i, c in enumerate(CTC_CHARS)}           # 1‑based → char

# ---------------------------------------------------------------------------
# GT util
# ---------------------------------------------------------------------------

def _decode_plate(idx_str: str) -> str | None:
    """"0_0_22_27_27_33_16" → "皖A04025" (None se parsing fallisce)."""
    try:
        seq = [int(x) for x in idx_str.split("_")]
        if len(seq) != 7:
            return None
        return (_PROVINCES[seq[0]] + _ALPHABETS[seq[1]] +
                "".join(_ADS[i] for i in seq[2:]).replace("O", ""))
    except (ValueError, IndexError):
        return None

# pattern: qualunque segmento con 6 "_" consecutivi e solo cifre → indice CCPD
IDX_RE = re.compile(r"(?:^|-)((?:\d+_){6}\d+)(?:-|$)")

def plate_from_filename(stem: str) -> str:
    """Estrae la targa GT dal nome *crop* (ritorna stringa vuota se fallisce)."""
    m = IDX_RE.search(stem)
    if m:
        p = _decode_plate(m.group(1))
        if p:
            return p
    return ""

lev = lambda a, b: SequenceMatcher(None, a, b).ratio()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CropDS(Dataset):
    def __init__(self, folder: pathlib.Path, tfm):
        self.paths = sorted(folder.glob("*.jpg"))
        self.tfm   = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p   = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.tfm(img), plate_from_filename(p.stem), p.name

# ---------------------------------------------------------------------------
# Greedy decode (per singola sequenza logit T×C)
# ---------------------------------------------------------------------------

def ctc_decode(seq: torch.Tensor | torch.Tensor) -> str:
    """`seq` shape (T, C). Ritorna stringa decodificata greedy‑CTC."""
    idxs = seq.argmax(1).cpu().numpy()  # (T,)
    out, prev = [], -1
    for v in idxs:
        if v != prev and v != 0:
            out.append(IDX2CHAR.get(int(v), "?"))
        prev = v
    return "".join(out)

# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser("OCR‑CTC evaluation")
    ap.add_argument("--data_root", required=True, type=pathlib.Path,
                    help="Cartella con crop .jpg")
    ap.add_argument("--weights",   required=True, type=pathlib.Path,
                    help="Checkpoint .pth da valutare")
    ap.add_argument("--batch", type=int, default=64, help="Batch size (inference)")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.data_root.exists():
        raise FileNotFoundError(args.data_root)
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)

    tfm = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor()
    ])
    dl = DataLoader(CropDS(args.data_root, tfm),
                    batch_size=args.batch, shuffle=False, num_workers=2)

    model = OCRModel(len(CTC_CHARS)).cuda().eval()
    model.load_state_dict(torch.load(args.weights, map_location="cuda"))

    rows, tot_sim = [], 0.0
    pbar = tqdm(dl, total=len(dl), desc="Val set")
    with torch.no_grad():
        for imgs, gts, fnames in pbar:
            logits = model(imgs.cuda()).log_softmax(2)            # B,W,C
            logits = logits.permute(1, 0, 2)                      # T,B,C
            B = imgs.size(0)
            for b in range(B):
                pred_str = ctc_decode(logits[:, b, :])
                gt_str   = gts[b]
                sim      = lev(pred_str, gt_str)
                tot_sim += sim
                rows.append({"file": fnames[b], "pred": pred_str,
                              "gt": gt_str, "lev": sim})
        pbar.close()

    out_csv = args.weights.parent / f"pred_{args.data_root.name}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n✅ CSV salvato → {out_csv.relative_to(pathlib.Path.cwd())} | "
          f"Levenshtein medio = {tot_sim/len(rows):.3f}")

if __name__ == "__main__":
    main()






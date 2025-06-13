# train.py – training CTC OCR baseline model usando **tokenizer CCPD**
# ----------------------------------------------------------------------------------------------------
"""Train su crop CCPD in cui la targa è **codificata nel nome‑file**.

*   VOCAB: province cinesi (34) + lettere (24) + cifre (10)
*   Per CTC usiamo **solo i token effettivi** (blank = 0).
*   Early‑stopping (patience 3) – **max 10 epoche**.
"""

import pathlib, re, torch, pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import OCRModel

# ───────────────────── CCPD TOKENIZER ─────────────────────
_PROVINCES = [
    "皖","沪","津","渝","冀","晋","辽","吉","黑","苏","浙","京","闽","赣","鲁",
    "豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新",
]
_ALPHABETS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
_ADS       = _ALPHABETS + list("0123456789") + ["O"]


def _decode_plate(idx_str: str) -> str | None:
    """'0_0_22_27_27_33_16'  →  '皖A04025' (None se parsing fallisce)."""
    try:
        arr = [int(x) for x in idx_str.split("_")]
        if len(arr) != 7:
            return None
        pro  = _PROVINCES[arr[0]]
        alp  = _ALPHABETS[arr[1]]
        tail = "".join(_ADS[i] for i in arr[2:]).replace("O", "")
        return pro + alp + tail
    except (ValueError, IndexError):
        return None

# ---- Token‑maps per CTC (escludiamo special token) --------------------------------
CTC_CHARS = _PROVINCES + _ALPHABETS + list("0123456789")
CHAR2IDX  = {c: i + 1 for i, c in enumerate(CTC_CHARS)}   # 0 = blank
IDX2CHAR  = {v: k for k, v in CHAR2IDX.items()}

# ───────────────────── CONFIG ─────────────────────
ROOT        = pathlib.Path(__file__).resolve().parents[1]
TRAIN_ROOT  = ROOT / "data" / "CCPD2019_crops" / "ccpd_base_crops"
VAL_ROOT    = ROOT / "data" / "CCPD2019_crops" / "ccpd_base_val_crops"
MODEL_PATH  = ROOT / "baseline" / "ocr_model.pth"
LOG_PATH    = ROOT / "baseline" / "training_log.csv"

# ───────────── helper per estrarre label dal filename ─────────────
IDX_RE = re.compile(r"^(?:.+-)*([0-9_]{5,})$")  # segmento di soli numeri/_ con almeno 5 num

def plate_from_filename(stem: str) -> str:
    """Cerca nel nome‑file il segmento con 6 underscore ⇒ 7 interi, quindi decodifica."""
    for part in stem.split("-"):
        if part.count("_") == 6 and part.replace("_", "").isdigit():
            dec = _decode_plate(part)
            if dec:
                return dec
    return ""

# ───────────── DATASET ─────────────────────
class PlateDataset(Dataset):
    def __init__(self, folder: str | pathlib.Path, transform=None):
        self.folder = pathlib.Path(folder)
        self.samples = []
        skipped = 0
        for img_path in sorted(self.folder.glob("*.jpg")):
            label = plate_from_filename(img_path.stem)
            if label:
                try:
                    enc = [CHAR2IDX[c] for c in label]
                    self.samples.append((img_path, enc))
                except KeyError:
                    skipped += 1
            else:
                skipped += 1
        if skipped:
            print(f"⚠️  {skipped} crop senza label utilizzabile in {self.folder.name} – ignorati")
        if not self.samples:
            raise RuntimeError(f"Nessuna immagine con label in {self.folder}")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, enc = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(enc, dtype=torch.long)

# ───────────── collate & decode ─────────────────────

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    seq_len  = imgs.shape[3] // 4   # larghezza   /4 ≈ tempo‑steps
    img_lens = torch.full((len(imgs),), seq_len, dtype=torch.long)
    lbl_lens = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels   = torch.cat(labels)
    return imgs, labels, img_lens, lbl_lens

# utile per debug
_decode_tensor = lambda t: "".join(IDX2CHAR.get(i.item(), "?") for i in t)

# ───────────── TRAINING ─────────────────────

def train(max_epochs: int = 10):
    tfm = transforms.Compose([
        transforms.Resize((32,128)),
        transforms.ToTensor(),
    ])
    train_ds = PlateDataset(TRAIN_ROOT, transform=tfm)
    val_ds   = PlateDataset(VAL_ROOT,   transform=tfm)

    train_ld = DataLoader(train_ds, batch_size=32, shuffle=True,  collate_fn=collate_fn)
    val_ld   = DataLoader(val_ds,   batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = OCRModel(len(CTC_CHARS)).cuda()
    ctc   = nn.CTCLoss(blank=0, zero_infinity=True)
    opt   = optim.Adam(model.parameters(), lr=1e-3)

    best_loss, patience, no_impr = float("inf"), 3, 0
    log = []
    print(f"\n🚀 Train su {TRAIN_ROOT.name} – max {max_epochs} epoche")

    for ep in range(1, max_epochs + 1):
        model.train(); total = 0.0
        for imgs, labels, img_lens, lbl_lens in tqdm(train_ld, desc=f"Ep {ep}"):
            imgs, labels = imgs.cuda(), labels.cuda()
            logits = model(imgs).log_softmax(2).permute(1,0,2)
            loss = ctc(logits, labels, img_lens, lbl_lens)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        avg_loss = total / len(train_ld)

        # ------------- quick validation -------------
        model.eval(); correct = chars = 0
        with torch.no_grad():
            for imgs, labels, img_lens, lbl_lens in val_ld:
                logits = model(imgs.cuda()).log_softmax(2).permute(1,0,2)
                pred   = logits.argmax(2)[:,0]
                prev = -1; dec = []
                for p in pred:
                    if p.item() != prev and p.item() != 0:
                        dec.append(IDX2CHAR.get(p.item(), '?'))
                    prev = p.item()
                target = _decode_tensor(labels[:lbl_lens[0]])
                correct += sum(p==t for p,t in zip(dec,target))
                chars   += len(target)
        acc = correct / chars if chars else 0
        print(f"📊 Ep {ep}: loss={avg_loss:.4f} | val_acc={acc:.4f}")
        log.append({"epoch": ep, "loss": avg_loss, "val_acc": acc})

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss; no_impr = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("  💾 nuovo best salvato")
        else:
            no_impr += 1
            if no_impr >= patience:
                print("⏹️  Early‑stop: nessun miglioramento per", patience, "epoche")
                break

    pd.DataFrame(log).to_csv(LOG_PATH, index=False)
    print(f"📝 Log scritto in {LOG_PATH.relative_to(ROOT)}")

if __name__ == "__main__":
    train()
















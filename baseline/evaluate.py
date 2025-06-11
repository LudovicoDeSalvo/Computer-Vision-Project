import os
import pathlib
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher

# === CONFIG ===
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "CCPD2019" / "images"
LABEL_DIR = ROOT / "data" / "CCPD2019" / "labels"
MODEL_PATH = ROOT / "baseline" / "ocr_model.pth"
OUTPUT_CSV = ROOT / "ctc_predictions.csv"
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 is reserved for CTC blank
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARS)}

# === DATASET ===
class PlateDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform=None):
        self.img_dir = pathlib.Path(img_dir)
        self.lbl_dir = pathlib.Path(lbl_dir)
        self.images = list(self.img_dir.glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        with open(lbl_path) as f:
            label_str = f.read().strip()
        label = [CHAR2IDX[c] for c in label_str if c in CHAR2IDX]
        return img, torch.tensor(label, dtype=torch.long)

# === MODEL ===
class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.rnn = nn.GRU(128 * 8, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, num_classes + 1)  # CTC requires +1 for blank

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, -1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# === GREEDY DECODING ===
def greedy_decode(logits):
    pred = logits.argmax(dim=2).detach().cpu().numpy()  # (T, B)
    pred = pred[:, 0] if pred.ndim == 2 else pred  # ensure shape (T,)
    prev = -1
    decoded = []
    for p in pred:
        if p != prev and p != 0:
            decoded.append(IDX2CHAR.get(int(p), '?'))
        prev = p
    return "".join(decoded)

# === EVALUATION ===
def evaluate():
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor()
    ])

    print("\n🚀 Avvio valutazione OCR CTC...")
    dataset = PlateDataset(DATA_DIR / "val", LABEL_DIR / "val", transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = OCRModel(len(CHARS)).cuda()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    results = []

    with torch.no_grad():
        for idx, (img, label) in enumerate(tqdm(loader, desc="🔍 Elaborazione immagini")):
            img = img.cuda()
            output = model(img)  # (B, T, C)
            output = output.log_softmax(2).permute(1, 0, 2)  # (T, B, C)
            pred_str = greedy_decode(output)
            target_str = "".join([IDX2CHAR.get(i.item(), '?') for i in label[0]])
            sim = SequenceMatcher(None, pred_str, target_str).ratio()
            results.append({"index": idx, "prediction": pred_str, "target": target_str, "levenshtein": sim})

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Risultati salvati in '{OUTPUT_CSV.name}'")
    print("📈 Similarità media Levenshtein:", df["levenshtein"].mean())

if __name__ == "__main__":
    evaluate()


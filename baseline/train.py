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
LOG_PATH = ROOT / "training_log.csv"
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
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        self.rnn = nn.GRU(256 * 4, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, num_classes + 1)  # CTC requires +1 for blank

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, -1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# === UTILITY ===
def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    img_lengths = torch.full(size=(len(imgs),), fill_value=imgs.shape[3] // 8, dtype=torch.long)  # width // pool
    label_lengths = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long)
    labels = torch.cat(labels)
    return imgs, labels, img_lengths, label_lengths

def decode_label(tensor):
    return "".join([IDX2CHAR.get(i.item(), '?') for i in tensor])

def levenshtein(a, b):
    return SequenceMatcher(None, a, b).ratio()

# === TRAINING ===
def train():
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor()
    ])

    train_dataset = PlateDataset(DATA_DIR / "train", LABEL_DIR / "train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = OCRModel(len(CHARS)).cuda()
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    log_data = []

    print("\n🚀 Avvio training OCR con CTC...")
    for epoch in range(10):
        model.train()
        epoch_loss = 0
        char_acc, total_chars = 0, 0

        for imgs, labels, img_lens, lbl_lens in tqdm(train_loader, desc=f"📚 Epoca {epoch+1}"):
            imgs, labels = imgs.cuda(), labels.cuda()
            logits = model(imgs)
            logits = logits.log_softmax(2).permute(1, 0, 2)
            loss = criterion(logits, labels, img_lens, lbl_lens)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Quick eval on first item in batch for char acc (optional)
            with torch.no_grad():
                pred = logits.argmax(2)[:, 0]  # (T,)
                pred_str = []
                prev = -1
                for p in pred:
                    if p.item() != prev and p.item() != 0:
                        pred_str.append(IDX2CHAR.get(p.item(), '?'))
                    prev = p.item()
                target_str = decode_label(labels[:lbl_lens[0]])
                char_acc += sum(p == t for p, t in zip(pred_str, target_str))
                total_chars += len(target_str)

        avg_loss = epoch_loss / len(train_loader)
        avg_acc = char_acc / total_chars if total_chars > 0 else 0
        print(f"✅ Fine epoca {epoch+1} | Loss media: {avg_loss:.4f} | Char Acc: {avg_acc:.4f}")
        log_data.append({"epoch": epoch+1, "loss": avg_loss, "char_acc": avg_acc})

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n💾 Modello salvato in '{MODEL_PATH.name}'")
    pd.DataFrame(log_data).to_csv(LOG_PATH, index=False)
    print(f"📝 Log di training salvato in '{LOG_PATH.name}'")

if __name__ == "__main__":
    train()



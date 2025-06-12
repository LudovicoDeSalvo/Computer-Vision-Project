"""
Train YOLOv5 on CCPD (cartelle standard images/labels)
───────────────────────────────────────────────────────
Directory attesa dal codice:
  data/CCPD2019/
    ├── images/train/*.jpg
    ├── images/val/*.jpg
    ├── images/test/*.jpg   (facoltativo)
    ├── labels/train/*.txt  (stessi nomi dei .jpg)
    ├── labels/val/*.txt
    └── labels/test/*.txt

Con questa struttura **non serve** specificare i percorsi label nel YAML:
YOLOv5 sostituisce automaticamente `images` → `labels` e `train/…` →
`train/…`. Il YAML quindi contiene solo i percorsi delle immagini.
"""

import subprocess, sys, pathlib, yaml, glob

# === Path setup ===
ROOT = pathlib.Path(__file__).resolve().parents[1]
YOLO = ROOT / "yolov5"
DATA_ROOT = ROOT / "data" / "CCPD2019"
IMG_ROOT = DATA_ROOT / "images"
LBL_ROOT = DATA_ROOT / "labels"

IMG_TRAIN = IMG_ROOT / "train"
IMG_VAL   = IMG_ROOT / "val"
IMG_TEST  = IMG_ROOT / "test"  # opzionale

# === YAML minimale ===
yaml_dict = {
    "train": str(IMG_TRAIN),
    "val":   str(IMG_VAL),
    # "test":  str(IMG_TEST),   # scommentare per split test
    "nc": 1,
    "names": ["plate"],
}
ABS_YAML = YOLO / "data" / "ccpd_abs.yaml"
ABS_YAML.write_text(yaml.dump(yaml_dict))
print("📄  YAML scritto in", ABS_YAML)

# === Check rapido ===
n_img = len(list(IMG_TRAIN.glob("*.jpg")))
n_lbl = len(list((LBL_ROOT / "train").glob("*.txt")))
if n_img == 0 or n_lbl == 0:
    sys.exit("❌ images/train o labels/train vuoti o non trovati.")
print(f"✅ {n_img} immagini e {n_lbl} label trovati per il training.")

# === Lancia training ===
cmd = [
    sys.executable, str(YOLO / "train.py"),
    "--data", str(ABS_YAML),
    "--img", "640",
    "--batch", "24",
    "--epochs", "30",
    "--freeze", "10",
    "--workers", "2",
    "--patience", "5",
    "--weights", str(YOLO / "yolov5s.pt"),
    "--name", "ccpd-yolov5s_fast"
]

print("\n⏳  Avvio training YOLOv5 …\n")
ret = subprocess.run(cmd, cwd=ROOT)
print("\n🏁  Training terminato con codice", ret.returncode)
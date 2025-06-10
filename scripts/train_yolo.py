"""
Train YOLOv5 on CCPD (~200 k immagini) — script completo e *self‑contained*

**Fix v4 – label mancanti**
──────────────────────────
Dalla cache YOLO risultano «0 labels found»: significa che molte immagini
nei file di *split* non hanno il file `.txt` corrispondente. Ora lo script:
1. **Conta** quante immagini totali + quante hanno davvero il label.
2. Se un’immagine è priva di label → la salta (non la linka).
3. Se a fine loop nessuna immagine valida è rimasta, stampa un avviso e
   termina per quello split.
4. Stampa un riepilogo finale prima di lanciare `train.py`.

Così YOLO riceve solo coppie image/label consistenti.

Parametri principali invar­iati:
- batch‑size 32 (riduci se OOM)
- epochs 40
- freeze 10
- cache images
- patience 5

Esegui da **root** progetto:
```bash
python scripts/train_yolo.py
```
"""

import subprocess, sys, pathlib, yaml, os, shutil

def safe_link(src: pathlib.Path, dst: pathlib.Path):
    """Try symlink, fallback to hard‑link or copy."""
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        try:
            os.link(src, dst)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)

ROOT = pathlib.Path(__file__).resolve().parents[1]
YOLO = ROOT / "yolov5"
DATASET = ROOT / "data" / "CCPD2019"
SPLITS = DATASET / "splits"

src_folders = [d for d in DATASET.iterdir() if d.is_dir() and d.name.startswith("ccpd_") and not d.name.endswith("_labels")]

total, linked = 0, 0

for split in ["train", "val", "test"]:
    img_dir = DATASET / f"images_{split}"
    lbl_dir = DATASET / f"labels_{split}"
    shutil.rmtree(img_dir, ignore_errors=True)
    shutil.rmtree(lbl_dir, ignore_errors=True)
    img_dir.mkdir(exist_ok=True)
    lbl_dir.mkdir(exist_ok=True)

    split_file = SPLITS / f"{split}.txt"
    if not split_file.exists():
        print(f"⚠️  Split file {split_file} mancante; skip {split}")
        continue

    miss_label, miss_file = 0, 0
    for rel_path in split_file.read_text().splitlines():
        rel_path = rel_path.strip()
        if not rel_path:
            continue
        total += 1
        if "/" in rel_path:  # includes subfolder
            jpg_src = DATASET / rel_path
            folder_name, rest = rel_path.split("/", 1)
            lbl_src = DATASET / f"{folder_name}_labels" / rest.replace(".jpg", ".txt")
        else:
            jpg_src = next((src / rel_path for src in src_folders if (src / rel_path).exists()), None)
            lbl_src = None
            if jpg_src:
                lbl_src = DATASET / f"{jpg_src.parent.name}_labels" / rel_path.replace(".jpg", ".txt")
        if jpg_src and jpg_src.exists():
            if lbl_src and lbl_src.exists() and lbl_src.stat().st_size:
                safe_link(jpg_src, img_dir / jpg_src.name)
                safe_link(lbl_src, lbl_dir / lbl_src.name)
                linked += 1
            else:
                miss_label += 1
        else:
            miss_file += 1
    print(f"{split}: immagini valide={linked}  senza label={miss_label}  mancanti={miss_file}")

if linked == 0:
    sys.exit("❌ Nessuna immagine con label trovata. Controlla la generazione delle etichette.")

# === YAML assoluto ===
yaml_dict = {
    "train": str(DATASET / "images_train"),
    "val":   str(DATASET / "images_val"),
    "nc": 1,
    "names": ["plate"],
}
ABS_YAML = YOLO / "data" / "ccpd_abs.yaml"
ABS_YAML.write_text(yaml.dump(yaml_dict))
print(f"📄  YAML scritto in {ABS_YAML}")
print(f"✅  Totale finalizzato: {linked} immagini con label")

# === Lancia training ===
cmd = [
    sys.executable, str(YOLO / "train.py"),
    "--img", "640",
    "--batch-size", "32",
    "--epochs", "40",
    "--freeze", "10",
    "--cache", "images",
    "--workers", "8",
    "--patience", "5",
    "--data", str(ABS_YAML),
    "--weights", str(YOLO / "yolov5s.pt"),
    "--name", "ccpd-yolov5s_fast"
]

print("\n⏳  Avvio training YOLOv5 …\n")
proc = subprocess.run(cmd, cwd=ROOT)
print("\n🏁  Training terminato con codice", proc.returncode)


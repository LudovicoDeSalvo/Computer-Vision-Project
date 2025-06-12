import random
import shutil
from pathlib import Path

def split_dataset(
    src_dir: str | Path,
    val_dir: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42
):
    src = Path(src_dir)
    dst = Path(val_dir)
    dst.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in src.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".jpeg"}]

    random.seed(seed)
    k = int(len(imgs) * val_ratio)
    val_imgs = random.sample(imgs, k)

    for img in val_imgs:
        shutil.move(str(img), str(dst / img.name))
    print(f"Moved {len(val_imgs)} images ({val_ratio*100:.1f}%) "
          f"from {src} → {dst}")

if __name__ == "__main__":
    split_dataset(
        src_dir="data/CCPD2019/ccpd_base",
        val_dir="data/CCPD2019/ccpd_base_val",
        val_ratio=0.1,
        seed=42
    )
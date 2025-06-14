import random
import shutil
from pathlib import Path
import argparse

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
    parser = argparse.ArgumentParser(description="Split image files from a source directory into a validation directory.")
    parser.add_argument("--src_dir", type=Path, required=True, help="Directory containing the source images.")
    parser.add_argument("--val_dir", type=Path, required=True, help="Directory where validation images will be moved.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of images to move to the validation set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    split_dataset(
        src_dir=args.src_dir,
        val_dir=args.val_dir,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
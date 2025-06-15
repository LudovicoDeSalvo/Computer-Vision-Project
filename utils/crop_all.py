import os
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Crop all CCPD2019 subsets using YOLO labels.")
    parser.add_argument("--repo_root", type=Path, required=True, help="Path to the repo root directory")
    args = parser.parse_args()

    folders = [
        "data/CCPD2019/ccpd_base",
        "data/CCPD2019/ccpd_base_val",
        "data/CCPD2019/ccpd_blur",
        "data/CCPD2019/ccpd_challenge",
        "data/CCPD2019/ccpd_db",
        "data/CCPD2019/ccpd_np",
        "data/CCPD2019/ccpd_fn",
        "data/CCPD2019/ccpd_rotate",
        "data/CCPD2019/ccpd_tilt",
        "data/CCPD2019/ccpd_weather",
    ]

    for rel_path in folders:
        image_dir = args.repo_root / rel_path
        tag = os.path.basename(image_dir)
        label_dir = args.repo_root / "data" / "CCPD2019" / f"{tag}_labels"
        output_dir = args.repo_root / "data" / "CCPD2019_crops" / f"{tag}_crops"

        print(f"\n🚀 Cropping {tag} → {output_dir} using labels from {label_dir}")

        cmd = [
            "python", str(args.repo_root / "utils" / "crop.py"),
            "--image_dir", str(image_dir),
            "--label_dir", str(label_dir),
            "--output_dir", str(output_dir),
        ]
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()

import os
import subprocess
import shutil
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv5 detection on CCPD folders and organize label outputs.")
    parser.add_argument("--repo_root", type=Path, required=True, help="Path to the repository root")
    args = parser.parse_args()

    # --- CONFIGURATION ---
    detect_script = args.repo_root / "yolov5" / "detect.py"
    weights = args.repo_root / "yolov5" / "runs" / "train" / "ccpd-yolov5s_fast15" / "weights" / "best.pt"
    img_size = 640
    conf_thres = 0.25
    project_dir = args.repo_root / "runs" / "detect"

    # --- MANUAL list of folders (relative to repo_root) ---
    rel_folders = [
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

    for rel_path in rel_folders:
        folder = args.repo_root / rel_path
        if not folder.is_dir():
            print(f"⚠️ Skipping (not a directory): {folder}")
            continue

        name = folder.name
        print(f"🔍 Detecting in: {folder}")

        # 1. Run detect.py
        cmd = [
            "python", str(detect_script),
            "--weights", str(weights),
            "--source", str(folder),
            "--img", str(img_size),
            "--conf", str(conf_thres),
            "--project", str(project_dir),
            "--name", name,
            "--exist-ok",
            "--save-txt",
            "--nosave"
        ]
        subprocess.run(cmd)

        # 2. Move labels
        label_src = project_dir / name / "labels"
        label_dst = args.repo_root / f"{rel_path}_labels"

        if label_src.exists():
            os.makedirs(label_dst, exist_ok=True)
            for f in os.listdir(label_dst):
                os.remove(label_dst / f)

            for file in os.listdir(label_src):
                shutil.move(label_src / file, label_dst / file)

            print(f"✅ Moved labels to: {label_dst}")
        else:
            print(f"⚠️ No labels found in {label_src} — detection may have failed.")

    print("🏁 All folders processed.")

if __name__ == "__main__":
    main()

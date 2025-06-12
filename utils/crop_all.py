import os
import subprocess

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

for image_dir in folders:
    tag = os.path.basename(image_dir)                       
    label_dir = f"data/CCPD2019/{tag}_labels"
    output_dir = f"data/CCPD2019_crops/{tag}_crops"

    print(f"\n🚀 Cropping {tag} → {output_dir} using labels from {label_dir}")

    
    cmd = [
        "python", "utils/crop.py",
        "--image_dir", image_dir,
        "--label_dir", label_dir,
        "--output_dir", output_dir,
    ]
    
    subprocess.run(cmd, check=True)
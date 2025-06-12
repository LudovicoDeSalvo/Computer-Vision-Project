import os
import subprocess
import shutil

# --- CONFIGURATION ---
detect_script = "yolov5/detect.py"
weights = "yolov5/runs/train/ccpd-yolov5s_fast15/weights/best.pt"
img_size = 640
conf_thres = 0.25
project_dir = "runs/detect"  # where YOLO puts its output by default

# --- MANUAL list of folders ---
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

# --- Run detection and move labels ---
for folder in folders:
    if not os.path.isdir(folder):
        print(f"⚠️ Skipping (not a directory): {folder}")
        continue

    name = os.path.basename(os.path.normpath(folder))
    print(f"🔍 Detecting in: {folder}")

    # 1. Run detect.py
    cmd = [
        "python", detect_script,
        "--weights", weights,
        "--source", folder,
        "--img", str(img_size),
        "--conf", str(conf_thres),
        "--project", project_dir,
        "--name", name,
        "--exist-ok",
        "--save-txt",  # <- important: tells YOLO to save labels
        "--nosave"
    ]
    subprocess.run(cmd)

    # 2. Move labels to the desired output
    label_src = os.path.join(project_dir, name, "labels")
    label_dst = f"{folder}_labels"

    if os.path.exists(label_src):
        os.makedirs(label_dst, exist_ok=True)
        for f in os.listdir(label_dst):
            os.remove(os.path.join(label_dst, f))

        for file in os.listdir(label_src):
            shutil.move(os.path.join(label_src, file), os.path.join(label_dst, file))

        print(f"✅ Moved labels to: {label_dst}")
    else:
        print(f"⚠️ No labels found in {label_src} — detection may have failed.")

print("🏁 All folders processed.")

import os, shutil, pathlib

root = "data/CCPD2019"
subsets = ["train", "val", "test"]          # cambia se hai solo train/val
img_folders = [f for f in os.listdir(root) if f.startswith("ccpd_") and not f.endswith("_labels")]

for split in subsets:
    split_dir = os.path.join(root, f"images_{split}")
    lbl_dir   = os.path.join(root, f"labels_{split}")
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    with open(os.path.join(root, "splits", f"{split}.txt")) as f:
        names = [line.strip() for line in f if line.strip()]

    for folder in img_folders:
        for name in names:
            img_src  = os.path.join(root, folder, name)                  # .jpg
            lbl_src  = os.path.join(root, folder + "_labels", name.replace(".jpg", ".txt"))
            if os.path.exists(img_src) and os.path.exists(lbl_src):
                # link simbolico 
                pathlib.Path(os.path.join(split_dir,  name)).symlink_to(os.path.relpath(img_src , split_dir))
                pathlib.Path(os.path.join(lbl_dir,   name.replace(".jpg", ".txt"))).symlink_to(
                    os.path.relpath(lbl_src, lbl_dir))
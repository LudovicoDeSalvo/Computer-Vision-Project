import os
from PIL import Image

def parse_filename_to_bbox(filename, img_w, img_h):
    try:
        # Esempio nome: 01-86_91-298&341_449&414-458&394_308&410_304&357_454&341-...
        coords = filename.split("-")[2]
        top_left, bottom_right = coords.split("_")
        x1, y1 = map(int, top_left.split("&"))
        x2, y2 = map(int, bottom_right.split("&"))

        # Calcola bbox normalizzata
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = abs(x2 - x1) / img_w
        height = abs(y2 - y1) / img_h

        return [0, x_center, y_center, width, height]
    except Exception as e:
        print(f"Errore parsing {filename}: {e}")
        return None

def generate_labels(image_dir, label_dir):
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    for fname in os.listdir(image_dir):
        if not fname.endswith(".jpg") or "-" not in fname:
            continue

        fpath = os.path.join(image_dir, fname)
        try:
            with Image.open(fpath) as img:
                w, h = img.size
                bbox = parse_filename_to_bbox(fname, w, h)
                if bbox:
                    label_path = os.path.join(label_dir, fname.replace(".jpg", ".txt"))
                    with open(label_path, "w") as f:
                        f.write(" ".join([str(round(x, 6)) for x in bbox]) + "\n")
        except Exception as e:
            print(f"Errore file {fname}: {e}")


if __name__ == "__main__":
    base_root = "data/CCPD2019"
    subfolders = [
        "ccpd_base", "ccpd_blur", "ccpd_challenge", "ccpd_db",
        "ccpd_fn", "ccpd_np", "ccpd_rotate", "ccpd_tilt", "ccpd_weather"
    ]

    for sf in subfolders:
        img_dir = os.path.join(base_root, sf)
        lbl_dir = os.path.join(base_root, sf + "_labels")
        print(f"🔄  Genero label per {sf} …")
        generate_labels(img_dir, lbl_dir)


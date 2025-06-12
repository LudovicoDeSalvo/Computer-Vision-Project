import os
import cv2
import glob
import argparse

def crop_from_labels(image_dir, label_dir, output_dir, img_ext=".jpg", crop_size=(144, 48)):
    os.makedirs(output_dir, exist_ok=True)
    total = 0
    missing = 0

    for img_path in glob.glob(os.path.join(image_dir, f"*{img_ext}")):
        base = os.path.basename(img_path)
        name, _ = os.path.splitext(base)
        label_path = os.path.join(label_dir, f"{name}.txt")

        if not os.path.exists(label_path):
            print(f"⚠️ Missing label for {img_path}")
            missing += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Could not read {img_path}")
            continue
        h, w, _ = img.shape

        with open(label_path, "r") as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, xc, yc, bw, bh = map(float, parts[:5])
                if int(cls) != 0:
                    continue  # skip non-plate class

                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, w), min(y2, h)

                crop = img[y1:y2, x1:x2]
                crop = cv2.resize(crop, crop_size)

                # keep original base name for cropped file
                out_filename = f"{name}_{i}.jpg"
                out_path = os.path.join(output_dir, out_filename)
                cv2.imwrite(out_path, crop)
                total += 1

    print(f"\n✅ Done. Saved {total} crops to {output_dir}. Skipped {missing} images with missing labels.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop license plates from YOLO label files")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to CCPD image directory")
    parser.add_argument("--label_dir", type=str, required=True, help="Path to YOLO label .txt files")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save cropped plates")
    parser.add_argument("--img_ext", type=str, default=".jpg", help="Image file extension")
    parser.add_argument("--crop_width", type=int, default=144, help="Width of cropped plate")
    parser.add_argument("--crop_height", type=int, default=48, help="Height of cropped plate")

    args = parser.parse_args()

    crop_from_labels(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        img_ext=args.img_ext,
        crop_size=(args.crop_width, args.crop_height),
    )

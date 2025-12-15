import os
import csv

ROOT_DIR = "ddata/test"
OUTPUT_CSV = "big_test.csv"

LABELS = {
    "real": 0,
    "fake": 1,
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

rows = []

for class_name, label in LABELS.items():
    class_dir = os.path.join(ROOT_DIR, class_name)
    if not os.path.isdir(class_dir):
        raise RuntimeError(f"Missing directory: {class_dir}")

    for root, _, files in os.walk(class_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMG_EXTS:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, ROOT_DIR)

                rows.append([rel_path, label])

print(f"Found {len(rows)} images")

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "label"])
    writer.writerows(rows)

print(f"Saved test CSV to {OUTPUT_CSV}")

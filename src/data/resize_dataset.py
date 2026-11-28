import csv
from pathlib import Path
from PIL import Image


# --------------------------------------------------
# 1. Paths (works both locally and on Kaggle)
# --------------------------------------------------

DATA_DIR = Path("../../data")
SPLIT_DIR = DATA_DIR / "splits"

# Target output directory (matches Kaggle requirement)
RESIZED_IMAGE_DIR = DATA_DIR / "dataset_256/images"
RESIZED_SPLIT_DIR = DATA_DIR / "dataset_256/splits"


# --------------------------------------------------
# 2. Resize helper (longer side = 256 px)
# --------------------------------------------------

def resize_longer_side(img, max_side=256):
    width, height = img.size

    if width >= height:
        new_width = max_side
        new_height = max(1, round(height * (max_side / width)))
    else:
        new_height = max_side
        new_width = max(1, round(width * (max_side / height)))

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


# --------------------------------------------------
# 3. Process one split
# --------------------------------------------------

def process_resized_split(split_name, max_side=256):
    input_csv = SPLIT_DIR / f"{split_name}.csv"

    with open(input_csv, "r", encoding="utf-8") as file:
        records = list(csv.DictReader(file))

    out_img_dir = RESIZED_IMAGE_DIR / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    RESIZED_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    updated_records = []

    print(f"\nResizing {split_name} split ({len(records)} images) to max side {max_side}px...")

    for idx, row in enumerate(records, start=1):
        src_path = DATA_DIR / row["image_path"]
        filename = src_path.name
        dst_path = out_img_dir / filename

        with Image.open(src_path) as img:
            rgb_img = img.convert("RGB")
            resized_img = resize_longer_side(rgb_img, max_side=max_side)
            resized_img.save(dst_path, format="JPEG", quality=92, optimize=True)

        updated_records.append({
            "image_path": f"images/{split_name}/{filename}",
            "image_url": row["image_url"],
            "class_label": row["class_label"],
            "label_idx": row["label_idx"]
        })

        if idx % 500 == 0 or idx == len(records):
            print(f"  Resized {idx}/{len(records)}")

    output_csv = RESIZED_SPLIT_DIR / f"{split_name}.csv"
    fieldnames = ["image_path", "image_url", "class_label", "label_idx"]

    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_records)

    print(f"Saved resized CSV: {output_csv}")


# --------------------------------------------------
# 4. Main execution
# --------------------------------------------------

def build_resized_dataset(max_side=256):
    for split_name in ["val", "test"]:
        process_resized_split(split_name, max_side=max_side)

    print("\n========== 3.3 RESIZE COMPLETE ==========")
    print(f"Output folder ready for Kaggle dataset upload: {DATA_DIR}")
    print("=========================================")


if __name__ == "__main__":
    build_resized_dataset(max_side=256)

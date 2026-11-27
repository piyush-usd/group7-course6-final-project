import csv
from pathlib import Path


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------
DATA_DIR = Path("../../data")
SPLIT_DIR = DATA_DIR / "splits"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Build CSV from already validated records
# --------------------------------------------------

def build_split_csv(downloaded_records, folder_name):

    csv_file = SPLIT_DIR / f"{folder_name}.csv"

    fieldnames = [
        "image_path",
        "image_url",
        "class_label",
        "label_idx"
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(downloaded_records)

    # Summary
    print(f"\n========== {folder_name.upper()} CSV REPORT ==========")
    print("Rows written to CSV:", len(downloaded_records))
    print("CSV created:", csv_file)
    print("=========================================")

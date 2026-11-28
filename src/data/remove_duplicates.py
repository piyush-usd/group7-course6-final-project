import csv
import numpy as np
import imagehash

from pathlib import Path
from PIL import Image
from collections import Counter
from create_csvs import build_split_csv


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

DATA_DIR = Path("../../data")
SPLIT_DIR = DATA_DIR / "splits"


# --------------------------------------------------
# 2. CSV Reader Helper
# --------------------------------------------------

def read_csv(csv_path):
    with open(csv_path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# --------------------------------------------------
# 3. Compute 64-bit perceptual hashes as uint64
# --------------------------------------------------

def compute_phashes(records, split_name):
    print(f"\nComputing 64-bit phashes for {split_name} ({len(records)} images)...")
    hashes = np.zeros(len(records), dtype=np.uint64)

    for idx, row in enumerate(records):
        img_path = DATA_DIR / row["image_path"]
        with Image.open(img_path) as img:
            ph = imagehash.phash(img.convert("RGB"), hash_size=8)
            hashes[idx] = np.uint64(int(str(ph), 16))

        if (idx + 1) % 500 == 0 or (idx + 1) == len(records):
            print(f"  Hashed {idx + 1}/{len(records)}")

    return hashes


# --------------------------------------------------
# 4. Chunked NumPy XOR + Popcount Duplicate Search
# --------------------------------------------------

POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def find_train_duplicates(train_hashes, query_hashes, threshold=4, chunk_size=512):
    is_duplicate = np.zeros(len(train_hashes), dtype=bool)

    for start in range(0, len(query_hashes), chunk_size):
        q_chunk = query_hashes[start : start + chunk_size]

        xor_matrix = train_hashes[:, None] ^ q_chunk[None, :]
        xor_bytes = xor_matrix.view(np.uint8).reshape(xor_matrix.shape + (8,))
        hamming_distances = POPCOUNT_LUT[xor_bytes].sum(axis=-1)

        is_duplicate |= (hamming_distances <= threshold).any(axis=1)

    return is_duplicate


# --------------------------------------------------
# 5. Main Deduplication Routine
# --------------------------------------------------

def deduplicate_train_split(threshold=4, chunk_size=512):
    train_records = read_csv(SPLIT_DIR / "train.csv")
    val_records = read_csv(SPLIT_DIR / "val.csv")
    test_records = read_csv(SPLIT_DIR / "test.csv")

    train_hashes = compute_phashes(train_records, "train")
    val_hashes = compute_phashes(val_records, "val")
    test_hashes = compute_phashes(test_records, "test")

    eval_hashes = np.concatenate([val_hashes, test_hashes])

    print(f"\nComparing {len(eval_hashes)} val/test hashes against {len(train_hashes)} train hashes...")
    dup_mask = find_train_duplicates(
        train_hashes=train_hashes,
        query_hashes=eval_hashes,
        threshold=threshold,
        chunk_size=chunk_size
    )

    cleaned_train = []
    removed_per_class = Counter()

    for row, is_dup in zip(train_records, dup_mask):
        if is_dup:
            removed_per_class[row["class_label"]] += 1
        else:
            cleaned_train.append(row)

    # Reuse build_split_csv from create_csvs.py to overwrite train.csv
    build_split_csv(cleaned_train, "train")

    # Summary Report
    print("\n========== 3.2 DEDUPLICATION REPORT ==========")
    print(f"Original train size : {len(train_records)}")
    print(f"Duplicates removed  : {int(dup_mask.sum())} (Hamming <= {threshold})")
    print(f"Cleaned train size  : {len(cleaned_train)}")
    print(f"Val size (intact)   : {len(val_records)}")
    print(f"Test size (intact)  : {len(test_records)}")

    if removed_per_class:
        print("\nDuplicates removed per class:")
        for label, count in sorted(removed_per_class.items()):
            print(f"  {label}: {count}")
    print("==============================================")


if __name__ == "__main__":
    deduplicate_train_split(threshold=4, chunk_size=512)

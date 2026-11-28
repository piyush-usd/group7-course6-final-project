import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from PIL import Image


# --------------------------------------------------
# 1. Project Paths
# --------------------------------------------------


DATA_DIR = Path("../../data")
SPLIT_DIR = DATA_DIR / "splits"

EDA_OUT_DIR = DATA_DIR / "reports" / "eda"

EDA_OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load All Splits
# --------------------------------------------------

def load_splits(split_dir=SPLIT_DIR):
    splits = {}
    for name in ["train", "val", "test"]:
        csv_path = Path(split_dir) / f"{name}.csv"
        if csv_path.exists():
            splits[name] = pd.read_csv(csv_path)
            print(f"{name.upper()}: {len(splits[name])} rows | {splits[name]['class_label'].nunique()} classes")
    return splits


# --------------------------------------------------
# 3. Class Counts per Split (Bar Chart)
# --------------------------------------------------

def plot_class_distribution(splits, save_dir=EDA_OUT_DIR):
    counts_df = pd.concat([
        df.assign(split=name) for name, df in splits.items()
    ], ignore_index=True)

    plt.figure(figsize=(14, 6))
    sns.countplot(data=counts_df, x="class_label", hue="split")
    plt.xticks(rotation=45, ha="right")
    plt.title("Class Distribution Across Train / Val / Test Splits")
    plt.xlabel("Class Label")
    plt.ylabel("Image Count")
    plt.tight_layout()

    out_path = Path(save_dir) / "01_class_distribution.png"
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved: {out_path}")


# --------------------------------------------------
# 4. Width, Height, Aspect Ratio & Background Brightness
# --------------------------------------------------

def plot_image_dimensions_and_brightness(train_df, sample_size=3000, save_dir=EDA_OUT_DIR):
    sample_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
    stats = []

    for _, row in sample_df.iterrows():
        img_path = DATA_DIR / row["image_path"]
        if not img_path.exists():
            continue

        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
            arr = np.asarray(rgb, dtype=np.float32) / 255.0

            # Sample outer 5px border to inspect background brightness
            border = np.concatenate([
                arr[:5, :, :].reshape(-1, 3),
                arr[-5:, :, :].reshape(-1, 3),
                arr[:, :5, :].reshape(-1, 3),
                arr[:, -5:, :].reshape(-1, 3),
            ], axis=0)

            stats.append({
                "width": w,
                "height": h,
                "aspect_ratio": w / h,
                "mean_brightness": arr.mean(),
                "border_brightness": border.mean(),
            })

    stats_df = pd.DataFrame(stats)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Aspect Ratio
    sns.histplot(stats_df["aspect_ratio"], bins=40, ax=axes[0], color="steelblue")
    axes[0].axvline(1.0, color="red", linestyle="--", label="Square (1:1)")
    axes[0].set_title("Aspect Ratio (Width / Height)")
    axes[0].legend()

    # Width & Height
    sns.histplot(stats_df["width"], bins=30, ax=axes[1], label="Width", alpha=0.6)
    sns.histplot(stats_df["height"], bins=30, ax=axes[1], label="Height", alpha=0.6)
    axes[1].set_title("Width & Height Distributions (px)")
    axes[1].legend()

    # Background Border Brightness
    sns.histplot(stats_df["border_brightness"], bins=30, ax=axes[2], color="goldenrod")
    axes[2].set_title("Border Background Brightness (0=Black, 1=White)")

    plt.tight_layout()
    out_path = Path(save_dir) / "02_dimensions_and_brightness.png"
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved: {out_path}")

    return stats_df


# --------------------------------------------------
# 5. 5 x N Sample Grid per Class
# --------------------------------------------------

def plot_sample_grid(train_df, n_samples=5, save_dir=EDA_OUT_DIR):
    classes = sorted(train_df["class_label"].unique())
    n_classes = len(classes)

    fig, axes = plt.subplots(
        n_classes,
        n_samples,
        figsize=(n_samples * 2.5, n_classes * 2.5)
    )

    for r, cls in enumerate(classes):
        cls_subset = train_df[train_df["class_label"] == cls]
        cls_rows = cls_subset.sample(n=min(n_samples, len(cls_subset)), random_state=42)

        for c, (_, row) in enumerate(cls_rows.iterrows()):
            ax = axes[r, c] if n_classes > 1 else axes[c]
            img_path = DATA_DIR / row["image_path"]
            if img_path.exists():
                with Image.open(img_path) as img:
                    ax.imshow(img.convert("RGB"))
            ax.axis("off")
            if c == 0:
                ax.set_title(cls, fontsize=10, loc="left", fontweight="bold")

    plt.tight_layout()
    out_path = Path(save_dir) / "03_sample_grid_per_class.png"
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved: {out_path}")


# --------------------------------------------------
# 6. Duplicates Summary Table
# --------------------------------------------------

def show_duplicate_table(dup_json_path=SPLIT_DIR / "duplicate_report.json"):
    dup_path = Path(dup_json_path)
    if not dup_path.exists():
        print(f"No duplicate report found at {dup_path}. Run Step 3.2 first or save removed_per_class.")
        return None

    with open(dup_path, "r", encoding="utf-8") as file:
        report = json.load(file)

    dup_df = pd.DataFrame.from_dict(
        report.get("removed_per_class", {}),
        orient="index",
        columns=["Duplicates Removed"]
    ).sort_index()
    dup_df.loc["TOTAL"] = dup_df["Duplicates Removed"].sum()

    print("\n========== DUPLICATE SUMMARY TABLE ==========")
    print(dup_df)
    print("=============================================")
    return dup_df


# --------------------------------------------------
# 7. Run Complete EDA Pipeline
# --------------------------------------------------

def run_eda():
    splits = load_splits()
    plot_class_distribution(splits)
    plot_image_dimensions_and_brightness(splits["train"])
    plot_sample_grid(splits["train"], n_samples=5)
    show_duplicate_table()


if __name__ == "__main__":
    run_eda()

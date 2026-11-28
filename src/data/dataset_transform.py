import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as T

from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------
# 1. Project Paths & Constants
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_DIR = DATA_DIR / "splits"

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet


# --------------------------------------------------
# 3.5 Pre-processing, Augmentation & Class Weights
# --------------------------------------------------

class LetterboxResize:
    def __init__(self, size, fill=(255, 255, 255)):
        self.size, self.fill = size, fill

    def __call__(self, img):
        img = ImageOps.contain(img, (self.size, self.size))
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(img, ((self.size - img.width) // 2, (self.size - img.height) // 2))
        return canvas


train_tf = T.Compose([
    LetterboxResize(240),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2, 0.02),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

eval_tf = T.Compose([
    LetterboxResize(240),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


def get_class_weights(train_csv=SPLIT_DIR / "train.csv"):
    """
    Computes inverse-frequency class weights w_c = N / (K * n_c)
    ordered by label_idx for CrossEntropyLoss(weight=...).
    """
    df = pd.read_csv(train_csv)
    counts = df["label_idx"].value_counts().sort_index()
    N = len(df)
    K = len(counts)
    weights = N / (K * counts.values.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------
# 3.7 Contour-Based Garment Crop
# --------------------------------------------------

def contour_crop(img_rgb, margin=0.05, min_area_frac=0.2):
    """Trims empty plain background around the garment using Canny + contours."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), None, iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return img_rgb

    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    H, W = img_rgb.shape[:2]

    if w * h < min_area_frac * H * W:  # contour too small -> keep full image
        return img_rgb

    dx, dy = int(margin * W), int(margin * H)
    return img_rgb[max(0, y - dy):min(H, y + h + dy), max(0, x - dx):min(W, x + w + dx)]


def contour_crop_pil(pil_img, margin=0.05, min_area_frac=0.2):
    """PIL wrapper around contour_crop so it plugs cleanly into GarmentDataset."""
    cropped_np = contour_crop(np.asarray(pil_img), margin=margin, min_area_frac=min_area_frac)
    return Image.fromarray(cropped_np)


def generate_contour_comparison(
    csv_path=SPLIT_DIR / "train.csv",
    out_path=PROJECT_ROOT / "reports" / "eda" / "contour_crop_before_after.png",
    n_samples=5,
    root_dir=PROJECT_ROOT
):
    """Generates the before/after contour crop deliverable figure for the report."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42)
    path_col = "image_path" if "image_path" in df.columns else "path"

    fig, axes = plt.subplots(2, len(sample_df), figsize=(len(sample_df) * 3, 6))

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        img_path = Path(root_dir) / row[path_col]
        orig_img = Image.open(img_path).convert("RGB")
        cropped_img = contour_crop_pil(orig_img)

        axes[0, idx].imshow(orig_img)
        axes[0, idx].set_title(f"Original\n({row['class_label']})", fontsize=9)
        axes[0, idx].axis("off")

        axes[1, idx].imshow(cropped_img)
        axes[1, idx].set_title("Contour Cropped", fontsize=9)
        axes[1, idx].axis("off")

    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.show()
    print(f"Saved before/after contour crop figure: {out_file}")


# --------------------------------------------------
# 3.6 Dataset and DataLoader
# --------------------------------------------------

class GarmentDataset(Dataset):
    def __init__(self, csv_path, transform, subset=None, crop_fn=None, root_dir=PROJECT_ROOT):
        df = pd.read_csv(csv_path)
        if subset:  # images per class, for fast early development
            df = df.groupby("label_idx", group_keys=False).head(subset)

        self.df = df.reset_index(drop=True)
        self.tf = transform
        self.crop = crop_fn
        self.root_dir = Path(root_dir)
        self.path_col = "image_path" if "image_path" in self.df.columns else "path"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = Path(row[self.path_col])
        if not img_path.is_absolute():
            img_path = self.root_dir / img_path

        img = Image.open(img_path).convert("RGB")
        if self.crop:
            img = self.crop(img)

        return self.tf(img), int(row.label_idx)


def get_dataloader(
    split,
    batch_size=64,
    train=False,
    subset=None,
    crop=False,
    num_workers=4,
    split_dir=SPLIT_DIR,
    root_dir=PROJECT_ROOT
):
    csv_path = Path(split_dir) / f"{split}.csv"
    transform = train_tf if train else eval_tf
    crop_fn = contour_crop_pil if crop else None

    dataset = GarmentDataset(
        csv_path=csv_path,
        transform=transform,
        subset=subset,
        crop_fn=crop_fn,
        root_dir=root_dir
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

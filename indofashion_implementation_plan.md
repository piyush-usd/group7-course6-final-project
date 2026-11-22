# IndoFashion Garment Classification — Implementation Plan

**Goal:** Classify Indian ethnic garments (saree, kurta, lehenga, etc.) from photos, comparing a classical baseline against several deep learning models.

**Dataset:** [Indo Fashion Dataset on Kaggle](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset) (~3 GB, ~106K images, 15 classes, official train / val / test split)
**Citation:** Rajput & Aneja, *IndoFashion: Apparel Classification for Indian Ethnic Clothes*, CVPR Workshops 2021 — https://indofashion.github.io
**Team:** 3 engineers, equal split, implementation only

---

## 1. Day 1 — Shared Setup (all three engineers)

Agree on these before splitting up so the pieces fit together later.

### 1.1 Environment

| Item | Choice |
|---|---|
| Compute | Kaggle Notebooks with GPU (each engineer uses their own account, so training runs in parallel) |
| Framework | PyTorch + `timm` (pretrained models) |
| Classical ML | `scikit-learn`, `scikit-image`, `opencv-python` |
| Utilities | `pandas`, `numpy`, `matplotlib`, `seaborn`, `imagehash`, `grad-cam` (imports as `pytorch_grad_cam`), `tabulate` |
| Code sharing | One GitHub repo; notebooks import from `src/` |

### 1.2 Repository layout

```
indofashion-classification/
├── configs/
│   └── default.yaml          # classes, image size, seed, paths
├── data/
│   └── splits/               # train.csv, val.csv, test.csv  (Engineer 1)
├── src/
│   ├── data/
│   │   ├── manifest.py       # JSON -> CSV, filtering, validation   (E1)
│   │   ├── dedup.py          # perceptual-hash duplicate check      (E1)
│   │   ├── transforms.py     # pre-processing + augmentation        (E1)
│   │   ├── crop.py           # contour-based garment crop           (E1)
│   │   └── dataset.py        # Dataset + get_dataloader()           (E1)
│   ├── features/
│   │   └── classical.py      # HOG + colour histogram features      (E1)
│   ├── models/
│   │   ├── small_cnn.py      # CNN from scratch                     (E2)
│   │   └── factory.py        # build_model(name, num_classes)       (E2/E3)
│   ├── train/
│   │   └── trainer.py        # shared training loop                 (E2)
│   └── eval/
│       ├── evaluate.py       # predict() + compute_metrics()        (E3)
│       ├── gradcam.py        # explainability                       (E3)
│       └── report.py         # comparison tables + charts           (E3)
├── notebooks/
│   ├── 01_eda.ipynb                  (E1)
│   ├── 02_classical_baseline.ipynb   (E1)
│   ├── 03_cnn_models.ipynb           (E2)
│   ├── 04_modern_model.ipynb         (E3)
│   └── 05_results.ipynb              (E3)
├── results/
│   └── <model_name>/         # best.pt, history.csv, metrics.json, figures
└── requirements.txt
```

### 1.3 Shared config (`configs/default.yaml`)

```yaml
seed: 42
image_size: 224
dataset_root: /kaggle/input/indo-fashion-dataset   # adjust to where Kaggle mounts it
splits_dir: data/splits
num_workers: 2

# Fill in from df["class_label"].unique() on Day 1.
# Suggested scope: women's garments only, footwear excluded.
include_classes:
  - saree
  - women_kurta          # placeholder names: use the exact strings from the JSON
  - leggings_and_salwar
  - palazzo
  - lehenga
  - dupattas
  - blouse
  - gowns
  - dhoti_pants
  - petticoats
```

### 1.4 Shared interfaces (the contract between workstreams)

| Interface | Owner | Signature | Used by |
|---|---|---|---|
| Split files | E1 | CSV columns: `path, label, label_idx` | Everyone |
| Data loader | E1 | `get_dataloader(split, batch_size=64, train=False, subset=None)` | E2, E3 |
| Training loop | E2 | `fit(model, train_dl, val_dl, cfg) -> history_df` | E2, E3 |
| Metrics | E3 | `compute_metrics(y_true, y_pred, y_score=None) -> dict` | Everyone (incl. SVM) |
| Results format | E3 | `results/<model>/metrics.json` | Final report |

**Unblocking tip:** E2 and E3 start on Day 2 with a quick `subset=` loader (e.g. 200 images per class) and switch to E1's final pipeline once it lands. `compute_metrics()` takes plain arrays, so the SVM baseline and the PyTorch models are scored identically.

---

## 2. Workstream Overview

| Rubric section | Engineer 1: Data & Classical | Engineer 2: CNN Models | Engineer 3: Modern Model & Evaluation |
|---|---|---|---|
| EDA & Pre-processing | Cleaning, EDA, pre-processing pipeline, contour crop | Shared training loop | Evaluation module |
| Modeling | HOG + colour histogram → SVM | Small CNN from scratch; ResNet-50 transfer learning | EfficientNet-B0 (ViT-B/16 optional) |
| Validation | 5-fold stratified CV for the baseline | Hyperparameter tuning on val set | Single final test-set run for all models |
| Results | EDA figures | Learning curves | Comparison tables, confusion matrices, Grad-CAM, error analysis |

---

## 3. Engineer 1 — Data Pipeline & Classical Baseline

### 3.1 Build the manifest (`data`)

The JSON files are line-delimited (one record per line), so read them with `lines=True`.

```python
import os
import pandas as pd
from PIL import Image

def load_split(root, name):
    df = pd.read_json(f"{root}/{name}_data.json", lines=True)
    df["path"] = df["image_path"].apply(lambda p: os.path.join(root, p))
    return df.rename(columns={"class_label": "label"})[["path", "label"]]

def is_valid(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False
```

Steps:
1. Load `train`, `val`, `test` JSONs.
2. Filter to `include_classes`; build `label_idx` from a sorted class list (save the mapping as `classes.json`).
3. Drop unreadable or missing images; log how many were dropped per class.
4. Write `data/splits/{train,val,test}.csv`.

### 3.2 Duplicate check (`data`)

Images were scraped from e-commerce sites, so the same product can appear in more than one split.

- Compute a 64-bit perceptual hash (`imagehash.phash`) for every image.
- Compare each val/test hash against all train hashes using NumPy XOR + popcount, in chunks.
- Treat Hamming distance ≤ 4 as a duplicate.
- **Remove duplicates from train, not test**, so the official test set stays intact and results remain comparable with the original paper.

### 3.3 Pre-resized copy (speed-up for everyone)

Decoding full-size JPEGs every epoch is the main bottleneck on Kaggle.
- Resize every image once so its longer side is 256 px, save to `/kaggle/working/`, and publish it as a private Kaggle dataset.
- E2 and E3 attach that dataset instead of the original.

### 3.4 EDA notebook (`notebooks/01_eda.ipynb`)

| Analysis | Output | Why it matters |
|---|---|---|
| Class counts per split | Bar chart | Shows train imbalance vs balanced val/test → justifies class weighting |
| Image width/height, aspect ratio | Histograms | Product photos are mostly tall portraits → justifies letterbox resizing over centre-crop |
| Sample grid per class | 5×N image grid | Shows visual similarity (saree vs lehenga, kurta vs gown) |
| Background colour / mean brightness | Histogram | Confirms plain backgrounds → justifies white padding and contour crop |
| Duplicates found | Count table | Data-quality evidence |

### 3.5 Pre-processing & augmentation (`data`)

Letterboxing keeps the whole garment visible; a centre-crop would cut off the hem of long garments like gowns and lehengas.

```python
from PIL import Image, ImageOps
import torchvision.transforms as T

class LetterboxResize:
    def __init__(self, size, fill=(255, 255, 255)):
        self.size, self.fill = size, fill
    def __call__(self, img):
        img = ImageOps.contain(img, (self.size, self.size))
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(img, ((self.size - img.width) // 2, (self.size - img.height) // 2))
        return canvas

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # ImageNet

train_tf = T.Compose([
    LetterboxResize(240), T.RandomCrop(224), T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2, 0.02), T.ToTensor(), T.Normalize(MEAN, STD),
])
eval_tf = T.Compose([
    LetterboxResize(240), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD),
])
```

**Class imbalance:** compute class weights `w_c = N / (K * n_c)` from the train split and pass them to `CrossEntropyLoss(weight=...)`. Expose them as `get_class_weights()`.

### 3.6 Dataset and loader (`data`)

```python
class GarmentDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform, subset=None, crop_fn=None):
        df = pd.read_csv(csv_path)
        if subset:  # images per class, for fast early development
            df = df.groupby("label_idx", group_keys=False).head(subset)
        self.df, self.tf, self.crop = df.reset_index(drop=True), transform, crop_fn
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.path).convert("RGB")
        if self.crop:
            img = self.crop(img)
        return self.tf(img), row.label_idx

def get_dataloader(split, batch_size=64, train=False, subset=None, crop=False): ...
```

### 3.7 Contour-based garment crop (`data`)

Covers the rubric's "additional image processing" requirement. Trims empty background around the garment.

```python
def contour_crop(img_rgb, margin=0.05, min_area_frac=0.2):
    gray = cv2.GaussianBlur(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), None, iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img_rgb
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    H, W = img_rgb.shape[:2]
    if w * h < min_area_frac * H * W:      # contour too small -> keep full image
        return img_rgb
    dx, dy = int(margin * W), int(margin * H)
    return img_rgb[max(0, y - dy):min(H, y + h + dy), max(0, x - dx):min(W, x + w + dx)]
```

Deliverable: a before/after figure for the report. E2 runs one ResNet-50 experiment with and without the crop to measure its effect.

### 3.8 Classical baseline (`src/features/classical.py`, `notebooks/02_classical_baseline.ipynb`)

```python
def extract_features(img_rgb_128):
    gray = cv2.cvtColor(img_rgb_128, cv2.COLOR_RGB2GRAY)
    hog_vec = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    hsv = cv2.cvtColor(img_rgb_128, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).flatten()
    return np.concatenate([hog_vec, hist / (hist.sum() + 1e-8)])   # ~2,300 dims

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=256, random_state=42)),
    ("svm", LinearSVC(C=0.1)),
])
```

- Images: letterboxed to 128×128.
- Train on a stratified subset (~1,000 per class); an SVM on the full 91K is too slow.
- Tune `C ∈ {0.01, 0.1, 1}` with `StratifiedKFold(n_splits=5)` via `GridSearchCV`, scoring `f1_macro`.
- Score on the full test set through E3's `compute_metrics()`, passing `decision_function` output as `y_score` for top-2 accuracy.

### Engineer 1 — Done when
- [ ] Split CSVs + `classes.json` committed
- [ ] Pre-resized dataset published on Kaggle
- [ ] `get_dataloader()` and `get_class_weights()` working
- [ ] EDA notebook with all figures
- [ ] Contour crop + before/after figure
- [ ] SVM baseline: CV results + `results/svm_hog/metrics.json`

---

## 4. Engineer 2 — CNN Models

### 4.1 Shared training loop (`src/train/trainer.py`)

```python
def fit(model, train_dl, val_dl, cfg):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device),
                                  label_smoothing=cfg.get("label_smoothing", 0.0))
    opt = torch.optim.AdamW(param_groups(model, cfg), weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda")
    best_f1, patience_left, history = -1, cfg.patience, []

    for epoch in range(cfg.epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        val = run_validation(model, val_dl)   # uses E3's compute_metrics()
        history.append({"epoch": epoch, **val})
        if val["macro_f1"] > best_f1:
            best_f1, patience_left = val["macro_f1"], cfg.patience
            torch.save(model.state_dict(), f"results/{cfg.name}/best.pt")
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    pd.DataFrame(history).to_csv(f"results/{cfg.name}/history.csv", index=False)
    return pd.DataFrame(history)
```

Features: mixed precision (AMP), early stopping on validation macro-F1, best-checkpoint saving, per-epoch history (train loss, val loss, val accuracy, val macro-F1), optional discriminative learning rates via `param_groups()`.

### 4.2 Model A — Small CNN from scratch (`src/models/small_cnn.py`)

```python
def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )

class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.head(self.features(x))
```

Purpose: show the gap between learning from scratch and transfer learning.

### 4.3 Model B — ResNet-50 transfer learning

```python
model = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
```

Two-stage fine-tuning:
1. **Stage 1 — head only:** freeze everything except `model.fc`; 2–3 epochs at LR 1e-3.
2. **Stage 2 — full fine-tune:** unfreeze all layers; backbone LR 1e-5, head LR 1e-4; up to 10 epochs with early stopping.

Extra experiment: Stage 2 with and without E1's contour crop.

### 4.4 Hyperparameter tuning

- Small grid on the **validation set only**: learning rate (3e-4, 1e-4, 3e-5) × augmentation strength (light vs. standard ColorJitter).
- Select by validation macro-F1; record every run in `results/tuning_log.csv`.

### 4.5 Outputs
- Learning curves (train vs. val loss, val macro-F1) for every model, plotted from `history.csv`.
- Best checkpoints for E3's final test run.

### Engineer 2 — Done when
- [ ] `fit()` working and used by E3
- [ ] Small CNN trained + learning curves
- [ ] ResNet-50 trained (both stages) + learning curves
- [ ] Contour-crop ablation result
- [ ] Tuning log committed

---

## 5. Engineer 3 — Modern Model & Evaluation

### 5.1 Model C — EfficientNet-B0 (ViT-B/16 optional)

```python
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
# optional, if GPU time allows:
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
```

- Trained with E2's `fit()`.
- EfficientNet-B0 is the primary choice: strong accuracy at low cost on Kaggle GPUs.
- ViT-B/16 needs a smaller LR with warmup and label smoothing 0.1 (see table below).

### 5.2 Starting hyperparameters (all models)

| Model | Batch | Optimizer | LR | Epochs (max) | Notes |
|---|---|---|---|---|---|
| SVM (HOG + colour) | — | LinearSVC | C ∈ {0.01, 0.1, 1} | — | 5-fold CV on subset |
| Small CNN | 64 | AdamW | 1e-3 | 20 | From scratch |
| ResNet-50 | 64 | AdamW | 1e-3 → 1e-4 / 1e-5 | 3 + 10 | Two-stage |
| EfficientNet-B0 | 64 | AdamW | 3e-4 | 10 | |
| ViT-B/16 (optional) | 32 | AdamW | 5e-5 + 1-epoch warmup | 8 | Label smoothing 0.1 |

These are starting points; E2's tuning grid refines them.

### 5.3 Evaluation module (`src/eval/evaluate.py`)

```python
@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, scores = [], []
    for x, y in loader:
        scores.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        ys.append(y)
    y_score = torch.cat(scores).numpy()
    return torch.cat(ys).numpy(), y_score.argmax(1), y_score

def compute_metrics(y_true, y_pred, y_score=None, class_names=None):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "per_class": classification_report(y_true, y_pred, target_names=class_names,
                                           output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None:
        m["top2_accuracy"] = top_k_accuracy_score(y_true, y_score, k=2)
    return m
```

Also recorded per model:
- **Parameter count:** `sum(p.numel() for p in model.parameters())`
- **Inference time:** mean ms/image at batch size 1 over 200 images, after 20 warm-up passes, with `torch.cuda.synchronize()` around timing.
- **Training time:** total wall-clock from E2's history.

Everything is written to `results/<model>/metrics.json`.

### 5.4 Explainability (`src/eval/gradcam.py`)

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

TARGET_LAYERS = {
    "small_cnn":       lambda m: [m.features[-1]],
    "resnet50":        lambda m: [m.layer4[-1]],
    "efficientnet_b0": lambda m: [m.conv_head],
}
cam = GradCAM(model=model, target_layers=TARGET_LAYERS[name](model))
heatmaps = cam(input_tensor=x)   # targets=None -> explains the predicted class
```

Deliverable: a grid of correct and incorrect predictions with heatmaps, showing whether each model focuses on the garment or on the background / model's face.

### 5.5 Error analysis
- Rank the off-diagonal confusion-matrix cells to find the most-confused class pairs (expected: saree ↔ lehenga, kurta ↔ gown).
- Grid of 16 misclassified test images with true label, predicted label, and confidence.

### 5.6 Final test-set run & report figures (`src/eval/report.py`, `notebooks/05_results.ipynb`)

**Test-set discipline:** the test set is used exactly once, after all tuning is frozen. All model selection uses the validation set.

Aggregate `results/*/metrics.json` into:

| Figure / table | Content |
|---|---|
| Model comparison table | Model, params, train time, inference ms/img, accuracy, macro-F1, top-2 accuracy |
| Bar chart | Accuracy and macro-F1 per model |
| Per-class F1 heatmap | Classes × models |
| Confusion matrices | One per model (normalised by row) |
| Accuracy vs. params scatter | Efficiency trade-off |
| Grad-CAM grid | From 5.4 |

Sanity reference: the original IndoFashion paper reports about 88% accuracy on all 15 classes with ResNet backbones.

### Engineer 3 — Done when
- [ ] `compute_metrics()` used by all models, including SVM
- [ ] EfficientNet-B0 trained (+ ViT if time allows)
- [ ] Grad-CAM grids for all CNN models
- [ ] Error analysis figures
- [ ] Final comparison table + all report figures

---

## 6. Milestones

| Milestone | Contents | Owner(s) |
|---|---|---|
| **M0 — Setup** | Repo, config, class list, interfaces agreed | All |
| **M1 — Data ready** | Split CSVs, pre-resized dataset, final `get_dataloader()`, EDA draft | E1 (E2/E3 work on subsets meanwhile) |
| **M2 — Models trained** | SVM, Small CNN, ResNet-50, EfficientNet-B0 checkpoints + learning curves | E1, E2, E3 in parallel |
| **M3 — Evaluation** | Single test-set run, metrics.json for every model, Grad-CAM, error analysis | E3 (inputs from all) |
| **M4 — Report assets** | All tables and figures mapped to rubric sections | All |

## 7. Rubric Mapping

| Rubric section | Primary owner | Artifacts |
|---|---|---|
| 1. Project Selection & Setup | All | Problem statement, dataset description, scope decision (classes kept) |
| 2. EDA & Pre-processing | E1 | EDA figures, cleaning/dedup stats, transforms, contour crop before/after |
| 3. Modeling Methods | E1, E2, E3 | SVM, Small CNN, ResNet-50, EfficientNet-B0 (+ ViT) implementations |
| 4. Validation & Metrics | E2, E3 | 5-fold CV (SVM), val-set tuning log, held-out test results, metric definitions |
| 5. Results & Findings | E3 | Comparison table, charts, confusion matrices, Grad-CAM, error analysis |

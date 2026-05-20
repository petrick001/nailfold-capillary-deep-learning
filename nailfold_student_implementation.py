"""
nailfold_student_implementation.py

Student implementation inspired by Bharathi et al. (2023):
"A deep learning system for quantitative assessment of microvascular abnormalities
in nailfold capillary images."

This is a simplified implementation for coursework.
I used the professors-provided folders:
- Segmentation.zip: native images + binary masks
- Density.zip: density images with 1 mm scale + native versions + count annotations

Main tasks implemented:
1. I Prepared segmentation dataset
2. Train U-Net for capillary/vessel segmentation
3. Predict segmentation masks and overlays
4. Calculate simple measurements from segmentation masks
5. Use density ground truth counts to compare estimated density/counts

Install:
    pip install torch torchvision opencv-python numpy pandas pillow tqdm matplotlib scikit-learn pymupdf

Typical use:
    python nailfold_student_implementation.py --mode prepare --seg_zip Segmentation.zip --density_zip Density.zip
    python nailfold_student_implementation.py --mode train --epochs 20
    python nailfold_student_implementation.py --mode predict
    python nailfold_student_implementation.py --mode density

Fast test:
    python nailfold_student_implementation.py --mode all --seg_zip Segmentation.zip --density_zip Density.zip --epochs 3
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}



# Basic utilities


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def unzip_file(zip_path: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def load_image_gray(path: Path, size: int = 256) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def load_mask_binary(path: Path, size: int = 256) -> np.ndarray:
    mask = Image.open(path).convert("L")
    mask = mask.resize((size, size), Image.NEAREST)
    arr = np.array(mask).astype(np.float32)
    arr = (arr > 127).astype(np.float32)
    return arr


def save_overlay(gray01: np.ndarray, mask01: np.ndarray, out_path: Path) -> None:
    base = (gray01 * 255).astype(np.uint8)
    base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    mask = (mask01 > 0.5).astype(np.uint8)
    colored = base_bgr.copy()
    colored[mask == 1] = (0, 255, 0)
    overlay = cv2.addWeighted(base_bgr, 0.65, colored, 0.35, 0)
    cv2.imwrite(str(out_path), overlay)



# Prepare professor data


def parse_density_ground_truth(pdf_path: Path, out_csv: Path) -> pd.DataFrame:
    """
    The GroundTruth.pdf contains names followed by capillary counts.
    Uses PyMuPDF if available; fallback to pdftotext command is not needed here.
    """
    text = ""
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(pdf_path))
        for page in doc:
            text += page.get_text("text") + "\n"
    except Exception:
        raise RuntimeError(
            "Could not read GroundTruth.pdf. Install pymupdf: pip install pymupdf"
        )

    tokens = [t.strip() for t in text.replace("\r", "\n").split() if t.strip()]
    names = [t for t in tokens if any(c.isalpha() for c in t) and any(c.isdigit() for c in t)]
    nums = [int(t) for t in tokens if t.isdigit()]

    n = min(len(names), len(nums))
    df = pd.DataFrame({"image_id": names[:n], "capillary_count_gt": nums[:n]})
    df.to_csv(out_csv, index=False)
    return df


def prepare(args) -> None:
    seed_everything(args.seed)
    work_dir = Path(args.work_dir)
    raw_dir = work_dir / "raw"
    prepared_dir = work_dir / "prepared"

    if args.clean and work_dir.exists():
        shutil.rmtree(work_dir)

    ensure_dir(raw_dir)
    ensure_dir(prepared_dir)

    if args.seg_zip:
        unzip_file(Path(args.seg_zip), raw_dir)
    if args.density_zip:
        unzip_file(Path(args.density_zip), raw_dir)

    seg_images = raw_dir / "Segmentation" / "seg_images"
    seg_masks = raw_dir / "Segmentation" / "seg_masks"

    if not seg_images.exists() or not seg_masks.exists():
        raise FileNotFoundError("Expected Segmentation/seg_images and Segmentation/seg_masks after unzipping.")

    pairs = []
    for img_path in list_images(seg_images):
        mask_path = seg_masks / f"{img_path.stem}mask{img_path.suffix}"
        if not mask_path.exists():
            candidates = list(seg_masks.glob(f"{img_path.stem}mask.*"))
            if candidates:
                mask_path = candidates[0]
            else:
                print(f"Warning: no mask found for {img_path.name}")
                continue
        pairs.append((img_path, mask_path))

    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    for split, split_pairs in splits.items():
        img_out = prepared_dir / "segmentation" / split / "images"
        mask_out = prepared_dir / "segmentation" / split / "masks"
        ensure_dir(img_out)
        ensure_dir(mask_out)
        for img_path, mask_path in split_pairs:
            shutil.copy2(img_path, img_out / img_path.name)
            # Rename mask to match image name. This makes training easier.
            shutil.copy2(mask_path, mask_out / img_path.name)

    density_pdf = raw_dir / "Density" / "GroundTruth.pdf"
    if density_pdf.exists():
        parse_density_ground_truth(density_pdf, prepared_dir / "density_ground_truth.csv")

    print("Preparation complete.")
    print(f"Segmentation pairs: {n}")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    print(f"Prepared data saved to: {prepared_dir}")



# Dataset


class SegmentationDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, size: int = 256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.images = list_images(image_dir)
        if not self.images:
            raise FileNotFoundError(f"No images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name
        img = load_image_gray(img_path, self.size)
        mask = load_mask_binary(mask_path, self.size)
        return (
            torch.from_numpy(img).unsqueeze(0).float(),
            torch.from_numpy(mask).unsqueeze(0).float(),
            img_path.name,
        )



# U-Net model


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)



# Loss and metrics


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        inter = (probs * targets).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1 - ((2 * inter + smooth) / (denom + smooth)).mean()
        return bce + dice_loss


def dice_iou(logits, targets, threshold=0.5) -> Tuple[float, float]:
    preds = (torch.sigmoid(logits) > threshold).float()
    smooth = 1e-6
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union_sum = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = ((2 * inter + smooth) / (union_sum + smooth)).mean().item()
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    iou = ((inter + smooth) / (union + smooth)).mean().item()
    return dice, iou



# Measurements


def clean_mask(mask01: np.ndarray) -> np.ndarray:
    mask = (mask01 > 0.5).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for lab in range(1, num):
        if stats[lab, cv2.CC_STAT_AREA] >= 15:
            cleaned[labels == lab] = 1
    return cleaned


def skeletonize(mask: np.ndarray) -> np.ndarray:
    img = (mask > 0).astype(np.uint8) * 255
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return (skel > 0).astype(np.uint8)


def calculate_measurements(mask01: np.ndarray) -> Dict[str, float]:
    """
    Simplified measurements calculated from the segmentation mask.
    These are approximate, not clinical-grade measurements.
    """
    mask = clean_mask(mask01)
    area_px = int(mask.sum())
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component_count = max(num - 1, 0)

    if area_px == 0 or component_count == 0:
        return {
            "vessel_area_px": 0,
            "component_count": 0,
            "density_per_100px": 0,
            "mean_width_px": 0,
            "max_width_px": 0,
            "shape_score": 0,
        }

    ys, xs = np.where(mask > 0)
    span = max(xs.max() - xs.min() + 1, 1)
    density_per_100px = component_count / span * 100

    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    skel = skeletonize(mask)
    widths = dist[skel > 0] * 2.0
    mean_width = float(widths.mean()) if len(widths) else 0.0
    max_width = float(widths.max()) if len(widths) else 0.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scores = []
    for c in contours:
        if cv2.contourArea(c) < 15:
            continue
        peri = cv2.arcLength(c, closed=True)
        x, y, w, h = cv2.boundingRect(c)
        diag = (w ** 2 + h ** 2) ** 0.5
        if diag > 0:
            scores.append(peri / diag)
    shape_score = float(np.mean(scores)) if scores else 0.0

    return {
        "vessel_area_px": area_px,
        "component_count": component_count,
        "density_per_100px": float(density_per_100px),
        "mean_width_px": mean_width,
        "max_width_px": max_width,
        "shape_score": shape_score,
    }


def estimate_scale_line_px(image_path: Path) -> float:
    """
    Density images include a 1 mm line. This function tries to detect the longest dark line.
    If detection fails, returns NaN.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return float("nan")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # black line is very dark compared to tissue background
    mask = cv2.inRange(gray, 0, 60)
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=40, minLineLength=80, maxLineGap=10)
    if lines is None:
        return float("nan")
    lengths = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line
        lengths.append(float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5))
    return max(lengths) if lengths else float("nan")



# Train, predict, density


def train(args) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    prepared = Path(args.work_dir) / "prepared" / "segmentation"
    train_ds = SegmentationDataset(prepared / "train" / "images", prepared / "train" / "masks", args.size)
    val_ds = SegmentationDataset(prepared / "val" / "images", prepared / "val" / "masks", args.size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet(base=args.base_channels).to(device)
    loss_fn = DiceBCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.work_dir) / "outputs"
    ensure_dir(out_dir)
    best_dice = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} train"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        val_dice, val_iou, val_loss = 0, 0, 0
        with torch.no_grad():
            for x, y, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} val"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += loss_fn(logits, y).item()
                d, i = dice_iou(logits, y)
                val_dice += d
                val_iou += i

        train_loss = total_loss / max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        val_dice /= max(len(val_loader), 1)
        val_iou /= max(len(val_loader), 1)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), out_dir / "unet_best.pth")
            print(f"Saved best model: {out_dir / 'unet_best.pth'}")

    print("Training done.")


def predict(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = Path(args.work_dir) / "prepared" / "segmentation"
    test_images = prepared / "test" / "images"
    test_masks = prepared / "test" / "masks"
    out_dir = Path(args.work_dir) / "outputs" / "segmentation_test"
    masks_out = out_dir / "predicted_masks"
    overlays_out = out_dir / "overlays"
    ensure_dir(masks_out)
    ensure_dir(overlays_out)

    model = UNet(base=args.base_channels).to(device)
    weights = Path(args.weights) if args.weights else Path(args.work_dir) / "outputs" / "unet_best.pth"
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    rows = []
    dataset = SegmentationDataset(test_images, test_masks, args.size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for x, y, names in tqdm(loader, desc="Predict test"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred = clean_mask((prob > args.threshold).astype(np.float32))

            d, i = dice_iou(logits, y, args.threshold)
            meas = calculate_measurements(pred)
            meas.update({"image": names[0], "dice": d, "iou": i})
            rows.append(meas)

            gray = x[0, 0].cpu().numpy()
            cv2.imwrite(str(masks_out / f"{Path(names[0]).stem}_mask.png"), (pred * 255).astype(np.uint8))
            save_overlay(gray, pred, overlays_out / f"{Path(names[0]).stem}_overlay.png")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "segmentation_test_results.csv", index=False)
    print(f"Saved results to {out_dir}")
    print(df[["image", "dice", "iou", "component_count", "mean_width_px", "max_width_px", "density_per_100px"]])


def density(args) -> None:
    """
    Applies the trained segmentation model to native density images and compares detected component count
    to ground-truth capillary counts from GroundTruth.pdf.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work = Path(args.work_dir)
    raw_density = work / "raw" / "Density" / "DATA"
    gt_csv = work / "prepared" / "density_ground_truth.csv"
    out_dir = work / "outputs" / "density"
    overlays_out = out_dir / "overlays"
    ensure_dir(overlays_out)

    if not gt_csv.exists():
        raise FileNotFoundError("Missing density_ground_truth.csv. Run --mode prepare first.")

    gt = pd.read_csv(gt_csv)

    model = UNet(base=args.base_channels).to(device)
    weights = Path(args.weights) if args.weights else work / "outputs" / "unet_best.pth"
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    rows = []
    with torch.no_grad():
        for _, r in tqdm(gt.iterrows(), total=len(gt), desc="Density evaluation"):
            image_id = str(r["image_id"])
            folder = raw_density / ''.join([c for c in image_id if not c.islower()])
            # Example image_id N1a -> folder N1. Easier fallback: search recursively.
            native_matches = list(raw_density.glob(f"**/{image_id}_Natif.*"))
            scale_matches = [p for p in raw_density.glob(f"**/{image_id}.*") if "_Natif" not in p.stem]
            if not native_matches:
                print(f"Warning: native image not found for {image_id}")
                continue
            native_path = native_matches[0]
            scale_path = scale_matches[0] if scale_matches else None

            gray = load_image_gray(native_path, args.size)
            x = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float().to(device)
            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred = clean_mask((prob > args.threshold).astype(np.float32))
            meas = calculate_measurements(pred)

            scale_px_per_mm = estimate_scale_line_px(scale_path) if scale_path else float("nan")
            gt_count = int(r["capillary_count_gt"])

            row = {
                "image_id": image_id,
                "native_image": str(native_path),
                "scale_image": str(scale_path) if scale_path else "",
                "gt_count": gt_count,
                "predicted_component_count": meas["component_count"],
                "absolute_count_error": abs(meas["component_count"] - gt_count),
                "gt_density_per_mm": gt_count / 1.0,
                "pred_density_per_100px": meas["density_per_100px"],
                "estimated_scale_px_per_mm": scale_px_per_mm,
                "mean_width_px": meas["mean_width_px"],
                "max_width_px": meas["max_width_px"],
                "shape_score": meas["shape_score"],
            }
            rows.append(row)
            save_overlay(gray, pred, overlays_out / f"{image_id}_overlay.png")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "density_results.csv", index=False)
    if not df.empty:
        print(f"Mean absolute count error: {df['absolute_count_error'].mean():.3f}")
    print(f"Saved density outputs to {out_dir}")
    print(df.head())



# Main


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prepare", "train", "predict", "density", "all"], required=True)
    p.add_argument("--seg_zip", default="Segmentation.zip")
    p.add_argument("--density_zip", default="Density.zip")
    p.add_argument("--work_dir", default="nailfold_work")
    p.add_argument("--weights", default="")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clean", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)
    elif args.mode == "density":
        density(args)
    elif args.mode == "all":
        prepare(args)
        train(args)
        predict(args)
        density(args)

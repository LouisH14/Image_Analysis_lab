"""
Dataset classes for UNO card DL classification pipeline.

Each game image is split into 5 area crops (center + 4 players).
Areas are labeled with multi-hot vectors over the 54 UNO card classes.
"""
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

# ─── Card vocabulary ─────────────────────────────────────────────────────────

CARD_CLASSES = [
    "b_0", "b_1", "b_2", "b_3", "b_4", "b_5", "b_6", "b_7", "b_8", "b_9",
    "b_draw_2", "b_reverse", "b_skip",
    "draw_4",
    "g_0", "g_1", "g_2", "g_3", "g_4", "g_5", "g_6", "g_7", "g_8", "g_9",
    "g_draw_2", "g_reverse", "g_skip",
    "r_0", "r_1", "r_2", "r_3", "r_4", "r_5", "r_6", "r_7", "r_8", "r_9",
    "r_draw_2", "r_reverse", "r_skip",
    "wild",
    "y_0", "y_1", "y_2", "y_3", "y_4", "y_5", "y_6", "y_7", "y_8", "y_9",
    "y_draw_2", "y_reverse", "y_skip",
]

CARD_TO_IDX = {c: i for i, c in enumerate(CARD_CLASSES)}
NUM_CLASSES = len(CARD_CLASSES)   # 54

# ─── Area geometry (mirrors core.py .area()) ─────────────────────────────────

_HEIGHT = 2662
_WIDTH  = 4000

def get_area_crop(img_array: np.ndarray, area_idx: int) -> np.ndarray:
    """Extract the region-of-interest for one position from a full table image."""
    h, w = img_array.shape[:2]
    if area_idx == 0:   # center card
        return img_array[h // 4 : h * 3 // 4, w // 4 : w * 3 // 4]
    if area_idx == 1:   # player 1 – bottom
        return img_array[h // 2 + 300 : h, 900 : w - 700]
    if area_idx == 2:   # player 2 – right
        return img_array[0 : h - 500, w - 1100 : w]
    if area_idx == 3:   # player 3 – top
        return img_array[0 : h // 2 - 300, 900 : w - 900]
    if area_idx == 4:   # player 4 – left
        return img_array[600 : h - 300, 0 : w // 2 - 600]
    raise ValueError(f"area_idx must be 0-4, got {area_idx}")


# ─── Label helpers ────────────────────────────────────────────────────────────

def parse_cards(cell: str) -> list[str]:
    """'r_3;b_skip' → ['r_3', 'b_skip'], 'EMPTY' → []."""
    if not cell or cell.strip() == "EMPTY":
        return []
    return [c.strip() for c in cell.split(";") if c.strip()]


def labels_to_multihot(cards: list[str]) -> torch.Tensor:
    vec = torch.zeros(NUM_CLASSES)
    for c in cards:
        if c in CARD_TO_IDX:
            vec[CARD_TO_IDX[c]] = 1.0
    return vec


# ─── Transforms ───────────────────────────────────────────────────────────────

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224) -> T.Compose:
    # Heavier augmentation is essential when training from scratch on a small dataset.
    return T.Compose([
        T.Resize((img_size + 48, img_size + 48)),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(30),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.08),
        T.RandomGrayscale(p=0.08),
        T.RandomPerspective(distortion_scale=0.3, p=0.4),
        T.ToTensor(),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        T.Normalize(IMG_MEAN, IMG_STD),
    ])


def get_val_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMG_MEAN, IMG_STD),
    ])


# ─── Datasets ─────────────────────────────────────────────────────────────────

class UNOAreaDataset(Dataset):
    """
    Multi-label dataset: one sample per (image, area) pair.

    Each sample is an area crop labeled with a 54-dim multi-hot vector.
    Center has exactly one 1; player areas have 1-3 ones (or all zeros for EMPTY).

    Args:
        csv_path:      Path to train.csv
        images_dir:    Folder containing the game JPEG images
        transform:     torchvision transform applied to each PIL crop
        include_empty: If True, include player areas labeled EMPTY (all-zero vector)
        indices:       Optional row indices to use (for train/val split)
    """

    def __init__(
        self,
        csv_path: str | Path,
        images_dir: str | Path,
        transform=None,
        include_empty: bool = False,
        indices: list[int] | None = None,
    ):
        self.transform = transform
        self.samples: list[tuple[Path, int, torch.Tensor]] = []

        images_dir = Path(images_dir)
        with open(csv_path) as f:
            rows = list(csv.reader(f))[1:]  # skip header

        if indices is not None:
            rows = [rows[i] for i in indices]

        for row in rows:
            img_id, center, _, p1, p2, p3, p4 = row
            img_path = self._find_image(images_dir, img_id)
            if img_path is None:
                continue

            area_labels = [
                parse_cards(center),
                parse_cards(p1),
                parse_cards(p2),
                parse_cards(p3),
                parse_cards(p4),
            ]
            for area_idx, cards in enumerate(area_labels):
                if not include_empty and not cards:
                    continue
                self.samples.append((img_path, area_idx, labels_to_multihot(cards)))

    @staticmethod
    def _find_image(folder: Path, img_id: str) -> Path | None:
        for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
            p = folder / f"{img_id}{ext}"
            if p.exists():
                return p
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, area_idx, label = self.samples[idx]
        img_array = np.array(Image.open(img_path).convert("RGB"))
        crop = Image.fromarray(get_area_crop(img_array, area_idx))
        if self.transform:
            crop = self.transform(crop)
        return crop, label


class UNOCenterDataset(Dataset):
    """
    Single-label dataset using only the center-card area.

    Simpler than UNOAreaDataset: guaranteed one card per sample → CrossEntropyLoss.
    Useful as a sanity check or for two-stage pipelines.
    """

    def __init__(
        self,
        csv_path: str | Path,
        images_dir: str | Path,
        transform=None,
        indices: list[int] | None = None,
    ):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        images_dir = Path(images_dir)
        with open(csv_path) as f:
            rows = list(csv.reader(f))[1:]

        if indices is not None:
            rows = [rows[i] for i in indices]

        for row in rows:
            img_id, center = row[0], row[1]
            if center not in CARD_TO_IDX:
                continue
            img_path = UNOAreaDataset._find_image(images_dir, img_id)
            if img_path is None:
                continue
            self.samples.append((img_path, CARD_TO_IDX[center]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img_array = np.array(Image.open(img_path).convert("RGB"))
        crop = Image.fromarray(get_area_crop(img_array, 0))  # area 0 = center
        if self.transform:
            crop = self.transform(crop)
        return crop, label


class UNOTemplateDataset(Dataset):
    """
    Single-label dataset built from individual card template images.

    Expects a folder where each file is named '<card_name>.jpg' (or .png).
    If the template folder is available, this provides clean per-class examples
    that can be combined with UNOAreaDataset via ConcatDataset.
    """

    def __init__(self, templates_dir: str | Path, transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        folder = Path(templates_dir)
        for f in sorted(folder.glob("*")):
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            name = f.stem
            if name in CARD_TO_IDX:
                self.samples.append((f, CARD_TO_IDX[name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

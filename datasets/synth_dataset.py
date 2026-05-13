from pathlib import Path
from typing import List, Tuple
import csv
import random

from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import numpy as np


def _sorted_card_names(src_crops_dir: Path) -> list[str]:
    crop_names = [
        p.name
        for p in src_crops_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
    ]
    return sorted(crop_names)


COLOR_NAMES = ["b", "g", "r", "y"]
COLOR_TO_IDX = {name: idx for idx, name in enumerate(COLOR_NAMES)}
RANK_NAMES = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "draw_2",
    "reverse",
    "skip",
]
RANK_TO_IDX = {name: idx for idx, name in enumerate(RANK_NAMES)}
SPECIAL_NAMES = ["wild", "draw_4"]
SPECIAL_TO_IDX = {name: idx for idx, name in enumerate(SPECIAL_NAMES)}
IGNORE_INDEX = -100


def _parse_card_name(name: str) -> dict:
    stem = Path(name).stem
    if stem in SPECIAL_TO_IDX:
        return {
            "kind": stem,
            "is_card": 1,
            "color": IGNORE_INDEX,
            "rank": IGNORE_INDEX,
            "special": SPECIAL_TO_IDX[stem],
        }

    parts = stem.split("_", 1)
    if len(parts) != 2:
        return {
            "kind": "unknown",
            "is_card": 0,
            "color": IGNORE_INDEX,
            "rank": IGNORE_INDEX,
            "special": IGNORE_INDEX,
        }

    color_name, rank_name = parts
    if color_name not in COLOR_TO_IDX or rank_name not in RANK_TO_IDX:
        return {
            "kind": "unknown",
            "is_card": 0,
            "color": IGNORE_INDEX,
            "rank": IGNORE_INDEX,
            "special": IGNORE_INDEX,
        }

    return {
        "kind": "colored",
        "is_card": 1,
        "color": COLOR_TO_IDX[color_name],
        "rank": RANK_TO_IDX[rank_name],
        "special": IGNORE_INDEX,
    }


def _build_useless_background(
    img_size: tuple[int, int], bg_color: tuple[int, int, int]
) -> Image.Image:
    canvas = Image.new("RGB", img_size, bg_color)
    draw = ImageDraw.Draw(canvas)

    width, height = img_size
    for _ in range(random.randint(3, 8)):
        x1 = random.randint(0, max(0, width - 1))
        y1 = random.randint(0, max(0, height - 1))
        x2 = min(width, x1 + random.randint(20, max(20, width // 3)))
        y2 = min(height, y1 + random.randint(20, max(20, height // 3)))
        color = tuple(random.randint(0, 255) for _ in range(3))
        shape = random.choice(["rectangle", "ellipse", "line"])
        if shape == "rectangle":
            draw.rectangle([x1, y1, x2, y2], outline=color, width=random.randint(1, 5))
        elif shape == "ellipse":
            draw.ellipse([x1, y1, x2, y2], outline=color, width=random.randint(1, 5))
        else:
            draw.line([x1, y1, x2, y2], fill=color, width=random.randint(1, 5))

    if random.random() < 0.5:
        noise = np.random.normal(0, 15, size=(height, width, 3)).astype(np.int16)
        base = np.array(canvas, dtype=np.int16)
        base = np.clip(base + noise, 0, 255).astype(np.uint8)
        canvas = Image.fromarray(base, mode="RGB")

    return canvas


class SynthDetectionDataset(Dataset):
    """Dataset for the generated large images and CSV placements.

    This loader resizes large images to a smaller training size and
    generates per-cell targets for a small detector.
    """

    def __init__(
        self,
        images_dir: str,
        placements_csv: str,
        src_crops_dir: str,
        img_size=(800, 532),
        grid_cell=16,
        transform=None,
        use_hsv_mask=False,
    ):
        self.images_dir = Path(images_dir)
        self.placements_csv = Path(placements_csv)
        self.src_crops_dir = Path(src_crops_dir)
        self.img_size = img_size
        self.grid_cell = grid_cell
        self.transform = transform
        self.use_hsv_mask = use_hsv_mask

        # mapping from crop name to (w,h) to compute centers
        self.crop_sizes = self._scan_crops()
        # parse placements CSV
        self.samples = self._read_csv()

    def _scan_crops(self):
        sizes = {}
        if self.src_crops_dir.exists():
            for p in self.src_crops_dir.iterdir():
                try:
                    im = Image.open(p)
                    sizes[p.name] = im.size
                    im.close()
                except Exception:
                    continue
        return sizes

    def _read_csv(self):
        samples = []
        with self.placements_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # group by big_image
            grouped = {}
            for row in reader:
                big = row["big_image"]
                grouped.setdefault(big, []).append(row)
            for big, rows in grouped.items():
                samples.append((big, rows))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        big_name, rows = self.samples[idx]
        img_path = self.images_dir / big_name
        img = Image.open(img_path).convert("RGB")

        # resize image and scale placements accordingly
        orig_w, orig_h = img.size
        tgt_w, tgt_h = self.img_size
        sx = tgt_w / orig_w
        sy = tgt_h / orig_h
        img = img.resize((tgt_w, tgt_h), resample=Image.BICUBIC)

        # Optional HSV saturation mask as an additional hint for the model
        hsv_mask = None
        if self.use_hsv_mask:
            hsv_img = img.convert("HSV")
            _, s, _ = hsv_img.split()
            # Saturation > 100 as requested
            s_np = np.array(s)
            hsv_mask = torch.from_numpy((s_np > 100).astype(np.float32)).unsqueeze(0)

        # feature map size estimation: depends on model downsample (train script expects div by 16)
        Hf = tgt_h // 16
        Wf = tgt_w // 16

        # prepare target tensors
        num_classes = len(self.crop_sizes) if self.crop_sizes else 54
        obj = torch.zeros((Hf, Wf), dtype=torch.float32)
        cls = torch.zeros((Hf, Wf), dtype=torch.long)
        offs = torch.zeros((Hf, Wf, 2), dtype=torch.float32)
        ang = torch.zeros((Hf, Wf, 2), dtype=torch.float32)

        for r in rows:
            cname = r["placed_image"]
            x = float(r["x"])
            y = float(r["y"])
            # if crop size known, compute center
            w, h = self.crop_sizes.get(cname, (250, 250))
            cx = x + w / 2.0
            cy = y + h / 2.0
            # scale
            cx *= sx
            cy *= sy
            # map to feature grid (downsample 16)
            gx = int(cx // 16)
            gy = int(cy // 16)
            if gx < 0 or gy < 0 or gx >= Wf or gy >= Hf:
                continue
            obj[gy, gx] = 1.0
            # class id as index in crop_sizes mapping (fallback random)
            try:
                cls_id = list(self.crop_sizes.keys()).index(cname)
            except Exception:
                cls_id = 0
            cls[gy, gx] = cls_id
            # offsets within cell (pixels)
            local_x = (cx - (gx * 16)) / 16.0
            local_y = (cy - (gy * 16)) / 16.0
            offs[gy, gx, 0] = local_x
            offs[gy, gx, 1] = local_y
            # angle
            angle = float(r.get("angle_deg", 0.0))
            rad = np.deg2rad(angle)
            ang[gy, gx, 0] = np.sin(rad)
            ang[gy, gx, 1] = np.cos(rad)

        img_t = TF.to_tensor(img)
        if self.use_hsv_mask and hsv_mask is not None:
            # Concatenate mask as a 4th channel
            img_t = torch.cat([img_t, hsv_mask], dim=0)

        if self.transform:
            img_t = self.transform(img_t)

        target = {
            "obj": obj,  # Hf x Wf
            "cls": cls,  # Hf x Wf
            "offs": offs,  # Hf x Wf x 2
            "ang": ang,  # Hf x Wf x 2
        }

        return img_t, target


class OnlineSynthDataset(Dataset):
    """Génère des images d'entraînement en mémoire sans utiliser le disque."""

    def __init__(
        self,
        src_crops_dir: str,
        img_size=(512, 512),
        grid_cell=16,
        per_image=1,
        use_hsv_mask=True,
        epoch_size=1000,
    ):
        self.src_crops_dir = Path(src_crops_dir)
        self.img_size = img_size
        self.grid_cell = grid_cell
        self.per_image = per_image
        self.use_hsv_mask = use_hsv_mask
        self.epoch_size = epoch_size
        self.bg_color = (205, 206, 210)

        self.crop_paths = [
            p
            for p in self.src_crops_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
        ]
        self.crop_names = [p.name for p in self.crop_paths]

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        # Créer le fond
        canvas = Image.new("RGB", self.img_size, self.bg_color)
        Hf, Wf = self.img_size[1] // self.grid_cell, self.img_size[0] // self.grid_cell

        obj = torch.zeros((Hf, Wf), dtype=torch.float32)
        offs = torch.zeros((Hf, Wf, 2), dtype=torch.float32)
        ang = torch.zeros((Hf, Wf, 2), dtype=torch.float32)

        for _ in range(self.per_image):
            crop_path = random.choice(self.crop_paths)
            item = Image.open(crop_path).convert("RGBA")

            # Rotation aléatoire
            angle = random.uniform(0, 360)
            rotated = item.rotate(angle, expand=True, resample=Image.BICUBIC)

            # Position aléatoire
            max_x = self.img_size[0] - rotated.width
            max_y = self.img_size[1] - rotated.height
            if max_x > 0 and max_y > 0:
                px, py = random.randint(0, max_x), random.randint(0, max_y)
                canvas.paste(rotated, (px, py), rotated)

                # Calcul du centre pour le label
                cx, cy = px + rotated.width / 2, py + rotated.height / 2
                gx, gy = int(cx // self.grid_cell), int(cy // self.grid_cell)

                if 0 <= gx < Wf and 0 <= gy < Hf:
                    obj[gy, gx] = 1.0
                    offs[gy, gx, 0] = (cx - (gx * self.grid_cell)) / self.grid_cell
                    offs[gy, gx, 1] = (cy - (gy * self.grid_cell)) / self.grid_cell
                    rad = np.deg2rad(angle)
                    ang[gy, gx, 0] = np.sin(rad)
                    ang[gy, gx, 1] = np.cos(rad)

        # Préparation des tenseurs
        img_t = TF.to_tensor(canvas)

        if self.use_hsv_mask:
            hsv_img = canvas.convert("HSV")
            _, s, _ = hsv_img.split()
            s_np = np.array(s)
            hsv_mask = torch.from_numpy((s_np > 100).astype(np.float32)).unsqueeze(0)
            img_t = torch.cat([img_t, hsv_mask], dim=0)

        return img_t, {"obj": obj, "offs": offs, "ang": ang}


class ClassificationSynthDataset(Dataset):
    """Synthetic dataset for hierarchical card classification.

    Samples are generated in memory at a fixed size. Negative samples are
    labeled ``useless``. Positive samples are split into colored cards and
    special cards with hierarchical labels:
    - card / useless
    - color among 4 families
    - rank among 13 values for colored cards
    - special among 2 values for wild / draw_4
    """

    def __init__(
        self,
        src_crops_dir: str,
        img_size=(512, 512),
        use_hsv_mask=True,
        epoch_size=1000,
        useless_prob=0.35,
        special_prob=0.1,
        save_debug_samples=False,
        debug_dir=None,
        debug_limit=8,
    ):
        self.src_crops_dir = Path(src_crops_dir)
        self.img_size = img_size
        self.use_hsv_mask = use_hsv_mask
        self.epoch_size = epoch_size
        self.useless_prob = useless_prob
        self.special_prob = special_prob
        self.save_debug_samples = save_debug_samples
        self.debug_dir = (
            Path(debug_dir)
            if debug_dir is not None
            else Path("scripts/vis_output/debug_samples")
        )
        self.debug_limit = debug_limit
        self.bg_color = (205, 206, 210)

        self.class_names = _sorted_card_names(self.src_crops_dir)
        self.card_meta = {name: _parse_card_name(name) for name in self.class_names}
        self.useless_name = "useless"
        self.useless_idx = len(self.class_names)
        self.num_classes = len(self.class_names) + 1
        self.num_color_classes = len(COLOR_NAMES)
        self.num_rank_classes = len(RANK_NAMES)
        self.num_special_classes = len(SPECIAL_NAMES)

        self.crop_paths = [
            p
            for p in self.src_crops_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
        ]
        self.crop_path_by_name = {p.name: p for p in self.crop_paths}

        self.colored_by_rank = {rank_name: [] for rank_name in RANK_NAMES}
        self.special_names = []
        for name, meta in self.card_meta.items():
            if meta["kind"] == "colored" and meta["rank"] != IGNORE_INDEX:
                self.colored_by_rank[RANK_NAMES[meta["rank"]]].append(name)
            elif meta["kind"] in SPECIAL_TO_IDX:
                self.special_names.append(name)

        self.available_ranks = [r for r, names in self.colored_by_rank.items() if names]

    def __len__(self):
        return self.epoch_size

    def _paste_random_card(self, canvas: Image.Image):
        crop_path = random.choice(self.crop_paths)
        crop_name = crop_path.name
        return self._paste_card(canvas, crop_path, crop_name)

    def _paste_specific_card(self, canvas: Image.Image, crop_name: str):
        crop_path = self.crop_path_by_name.get(crop_name)
        if crop_path is None:
            return self._paste_random_card(canvas)
        return self._paste_card(canvas, crop_path, crop_name)

    def _paste_card(self, canvas: Image.Image, crop_path: Path, crop_name: str):
        card = Image.open(crop_path).convert("RGBA")

        angle = random.uniform(0, 360)
        scale = random.uniform(0.85, 1.15)
        new_w = max(8, int(card.width * scale))
        new_h = max(8, int(card.height * scale))
        card = card.resize((new_w, new_h), resample=Image.BICUBIC)
        card = card.rotate(angle, expand=True, resample=Image.BICUBIC)

        # If rotation made the card larger than the canvas, downscale to fit
        img_w, img_h = self.img_size
        if card.width > img_w or card.height > img_h:
            factor = (
                min(img_w / (card.width + 1e-6), img_h / (card.height + 1e-6)) * 0.9
            )
            if factor <= 0:
                return canvas, self.useless_idx, self.useless_name
            new_w2 = max(8, int(card.width * factor))
            new_h2 = max(8, int(card.height * factor))
            card = card.resize((new_w2, new_h2), resample=Image.BICUBIC)

        max_x = self.img_size[0] - card.width
        max_y = self.img_size[1] - card.height
        if max_x < 0 or max_y < 0:
            return canvas, self.useless_idx, self.useless_name

        px = random.randint(0, max_x)
        py = random.randint(0, max_y)
        canvas.paste(card, (px, py), card)
        return canvas, crop_name

    def _label_text(self, meta: dict) -> str:
        if meta["is_card"] == 0:
            return self.useless_name
        if meta["special"] != IGNORE_INDEX:
            return SPECIAL_NAMES[meta["special"]]
        return f'{COLOR_NAMES[meta["color"]]}_{RANK_NAMES[meta["rank"]]}.png'

    def _save_debug_image(self, img: Image.Image, label_name: str, idx: int):
        if not self.save_debug_samples or idx >= self.debug_limit:
            return

        self.debug_dir.mkdir(parents=True, exist_ok=True)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 220, 34], fill=(0, 0, 0))
        draw.text((8, 8), f"label: {label_name}", fill=(255, 255, 255))
        img.save(self.debug_dir / f"sample_{idx:04d}_{label_name}.jpg")

    def __getitem__(self, idx):
        use_useless = random.random() < self.useless_prob
        if use_useless:
            canvas = _build_useless_background(self.img_size, self.bg_color)
            target = {
                "is_card": torch.tensor(0, dtype=torch.long),
                "color": torch.tensor(IGNORE_INDEX, dtype=torch.long),
                "rank": torch.tensor(IGNORE_INDEX, dtype=torch.long),
                "special": torch.tensor(IGNORE_INDEX, dtype=torch.long),
            }
            label_name = self.useless_name
        else:
            canvas = Image.new("RGB", self.img_size, self.bg_color)
            # Rank-priority sampling:
            # - special cards sampled via special_prob
            # - colored cards sampled with uniform rank selection first
            if self.special_names and random.random() < self.special_prob:
                crop_name = random.choice(self.special_names)
                canvas, crop_name = self._paste_specific_card(canvas, crop_name)
            elif self.available_ranks:
                rank_name = random.choice(self.available_ranks)
                crop_name = random.choice(self.colored_by_rank[rank_name])
                canvas, crop_name = self._paste_specific_card(canvas, crop_name)
            else:
                canvas, crop_name = self._paste_random_card(canvas)

            meta = self.card_meta.get(
                crop_name,
                {
                    "kind": "unknown",
                    "is_card": 0,
                    "color": IGNORE_INDEX,
                    "rank": IGNORE_INDEX,
                    "special": IGNORE_INDEX,
                },
            )
            target = {
                "is_card": torch.tensor(meta["is_card"], dtype=torch.long),
                "color": torch.tensor(meta["color"], dtype=torch.long),
                "rank": torch.tensor(meta["rank"], dtype=torch.long),
                "special": torch.tensor(meta["special"], dtype=torch.long),
            }
            label_name = self._label_text(meta)

        self._save_debug_image(canvas.copy(), label_name, idx)

        img_t = TF.to_tensor(canvas)

        if self.use_hsv_mask:
            hsv_img = canvas.convert("HSV")
            _, s, _ = hsv_img.split()
            s_np = np.array(s)
            hsv_mask = torch.from_numpy((s_np > 100).astype(np.float32)).unsqueeze(0)
            img_t = torch.cat([img_t, hsv_mask], dim=0)

        return img_t, target

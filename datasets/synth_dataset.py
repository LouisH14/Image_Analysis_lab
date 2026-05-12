from pathlib import Path
from typing import List, Tuple
import csv
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import numpy as np


class SynthDetectionDataset(Dataset):
    """Dataset for the generated large images and CSV placements.

    This loader resizes large images to a smaller training size and
    generates per-cell targets for a small detector.
    """

    def __init__(self, images_dir: str, placements_csv: str, src_crops_dir: str,
                 img_size=(800, 532), grid_cell=16, transform=None):
        self.images_dir = Path(images_dir)
        self.placements_csv = Path(placements_csv)
        self.src_crops_dir = Path(src_crops_dir)
        self.img_size = img_size
        self.grid_cell = grid_cell
        self.transform = transform

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
        with self.placements_csv.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # group by big_image
            grouped = {}
            for row in reader:
                big = row['big_image']
                grouped.setdefault(big, []).append(row)
            for big, rows in grouped.items():
                samples.append((big, rows))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        big_name, rows = self.samples[idx]
        img_path = self.images_dir / big_name
        img = Image.open(img_path).convert('RGB')

        # resize image and scale placements accordingly
        orig_w, orig_h = img.size
        tgt_w, tgt_h = self.img_size
        sx = tgt_w / orig_w
        sy = tgt_h / orig_h
        img = img.resize((tgt_w, tgt_h), resample=Image.BICUBIC)

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
            cname = r['placed_image']
            x = float(r['x'])
            y = float(r['y'])
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
            angle = float(r.get('angle_deg', 0.0))
            rad = np.deg2rad(angle)
            ang[gy, gx, 0] = np.sin(rad)
            ang[gy, gx, 1] = np.cos(rad)

        img_t = TF.to_tensor(img)
        if self.transform:
            img_t = self.transform(img_t)

        target = {
            'obj': obj,        # Hf x Wf
            'cls': cls,        # Hf x Wf
            'offs': offs,      # Hf x Wf x 2
            'ang': ang,        # Hf x Wf x 2
        }

        return img_t, target

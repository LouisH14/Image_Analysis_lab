#!/usr/bin/env python3
"""
Génère des images 4000x2662 en plaçant aléatoirement images
depuis `data/images_crop`. Fond uni (hex) par défaut `#cdced2`.
Enregistre chaque grande image dans `data/images_generee/` et écrit
un CSV global `placements_TIMESTAMP.csv` listant tous les placements.

Usage:
  python labs/generate_large_images.py --count 5
"""

from pathlib import Path
import argparse
import random
import csv
import time
from PIL import Image

CANVAS_W = 4000
CANVAS_H = 2662


def load_source_images(src_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if not src_dir.exists():
        return []
    imgs = [p for p in sorted(src_dir.iterdir()) if p.suffix.lower() in exts]
    return imgs


def choose_position(canvas_w, canvas_h, img_w, img_h):
    if img_w > canvas_w or img_h > canvas_h:
        return None
    x = random.randint(0, canvas_w - img_w)
    y = random.randint(0, canvas_h - img_h)
    return x, y


def _hex_to_rgb(hexcol: str):
    hexcol = hexcol.lstrip("#")
    if len(hexcol) == 6:
        r = int(hexcol[0:2], 16)
        g = int(hexcol[2:4], 16)
        b = int(hexcol[4:6], 16)
        return (r, g, b)
    raise ValueError("Invalid hex color")


def generate_one(
    output_dir: Path,
    images,
    num_paste=10,
    seed=None,
    out_name_prefix=None,
    bg_color: str = "#cdced2",
):
    if seed is not None:
        random.seed(seed + (hash(out_name_prefix) & 0xFFFFFFFF))

    rgb = _hex_to_rgb(bg_color) if bg_color else (205, 206, 210)
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (rgb[0], rgb[1], rgb[2], 255))

    records = []

    for i in range(num_paste):
        src = random.choice(images)
        im = Image.open(src).convert("RGBA")
        angle = random.uniform(0, 360)

        rotated = im.rotate(angle, expand=True, resample=Image.BICUBIC)
        if rotated.width > CANVAS_W or rotated.height > CANVAS_H:
            rotated = im.rotate(angle, expand=False, resample=Image.BICUBIC)

        pos = choose_position(CANVAS_W, CANVAS_H, rotated.width, rotated.height)
        if pos is None:
            pos = (0, 0)

        canvas.paste(rotated, pos, rotated)
        records.append((src.name, pos[0], pos[1], round(angle, 3)))

    timestamp = int(time.time())
    if out_name_prefix:
        base = f"{out_name_prefix}_{timestamp}"
    else:
        base = f"gen_{timestamp}"

    out_img_path = output_dir / f"{base}.jpg"
    canvas.convert("RGB").save(out_img_path, quality=95)

    return out_img_path, records


def main():
    p = argparse.ArgumentParser(
        description="Générer des images 4000x2662 à partir de crops"
    )
    p.add_argument(
        "--src_dir",
        default="data/images_crop",
        help="Dossier source des images à coller",
    )
    p.add_argument("--out_dir", default="data/images_generee", help="Dossier de sortie")
    p.add_argument("--count", type=int, default=1, help="Nombre d'images générées")
    p.add_argument(
        "--per_image",
        type=int,
        default=10,
        help="Nombre d'images collées par grande image",
    )
    p.add_argument("--seed", type=int, default=None, help="Seed aléatoire (optionnel)")
    p.add_argument(
        "--bg_color", default="#cdced2", help="Couleur de fond hex (ex: #cdced2)"
    )
    args = p.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = load_source_images(src_dir)
    if not images:
        print(f"Aucune image trouvée dans {src_dir}.")
        return

    timestamp_all = int(time.time())
    csv_path = out_dir / f"placements_{timestamp_all}.csv"
    print(f"Found {len(images)} source images. Generating {args.count} image(s)...")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["big_image", "placed_image", "x", "y", "angle_deg"])

        for i in range(args.count):
            prefix = f"gen{i}"
            img_path, records = generate_one(
                out_dir,
                images,
                num_paste=args.per_image,
                seed=args.seed,
                out_name_prefix=prefix,
                bg_color=args.bg_color,
            )
            for r in records:
                writer.writerow([img_path.name, r[0], r[1], r[2], r[3]])
            print(f"Wrote {img_path}")
    print(f"Placements CSV: {csv_path}")


if __name__ == "__main__":
    main()

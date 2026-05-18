#!/usr/bin/env python3
"""
Génère des images 4000x2662 en plaçant aléatoirement des images
depuis `data/images_crop`. Alterne 50/50 entre un fond uni (hex)
et un fond image depuis `data/colorfull_background`.
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
BACKGROUND_IMAGE_PROBABILITY = 0.5


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
    bg_images=None,
    make_grayscale: bool = False,
    token_image: Path = None,
):
    if seed is not None:
        random.seed(seed + (hash(out_name_prefix) & 0xFFFFFFFF))

    use_bg_image = bool(bg_images) and random.random() < BACKGROUND_IMAGE_PROBABILITY
    if use_bg_image:
        bg_path = random.choice(bg_images)
        with Image.open(bg_path) as bg_im:
            canvas = (
                bg_im.convert("RGB")
                .resize((CANVAS_W, CANVAS_H), resample=Image.LANCZOS)
                .convert("RGBA")
            )
    else:
        rgb = _hex_to_rgb(bg_color) if bg_color else (205, 206, 210)
        canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (rgb[0], rgb[1], rgb[2], 255))

    records = []

    for i in range(num_paste):
        src = random.choice(images)
        with Image.open(src) as im:
            im = im.convert("RGBA")
        angle = random.uniform(0, 360)

        rotated = im.rotate(angle, expand=True, resample=Image.BICUBIC)
        if rotated.width > CANVAS_W or rotated.height > CANVAS_H:
            rotated = im.rotate(angle, expand=False, resample=Image.BICUBIC)

        pos = choose_position(CANVAS_W, CANVAS_H, rotated.width, rotated.height)
        if pos is None:
            pos = (0, 0)

        canvas.paste(rotated, pos, rotated)
        records.append((src.name, pos[0], pos[1], round(angle, 3)))
    
    
    # Paste token image if provided
    if token_image and token_image.exists():
        if token_image.is_dir():
            # If it's a directory, pick one image randomly from it
            possible_tokens = load_source_images(token_image)
            actual_token = random.choice(possible_tokens) if possible_tokens else None
        else:
            actual_token = token_image

        if actual_token:
            with Image.open(actual_token) as sp_im:
                sp_im = sp_im.convert("RGBA")
            # For the token image, we use a fixed 0 degree rotation for now
            angle = 0.0
            pos = choose_position(CANVAS_W, CANVAS_H, sp_im.width, sp_im.height)
            if pos is None:
                pos = (0, 0)
            
            canvas.paste(sp_im, pos, sp_im)
            records.append((actual_token.name, pos[0], pos[1], round(angle, 3)))

    if make_grayscale:
        canvas = canvas.convert("L")

    timestamp = int(time.time())
    if out_name_prefix:
        base = f"{out_name_prefix}_{timestamp}"
    else:
        base = f"gen_{timestamp}"

    out_img_path = output_dir / f"{base}.jpg"
    canvas.convert("RGB").save(out_img_path, quality=95)

    return out_img_path, records


def main():
    # Calculate the project root directory (one level up from this script's directory)
    project_root = Path(__file__).resolve().parent.parent
    
    # Define paths relative to the project root
    src_path = project_root / "data" / "images_crop" 
    bg_path = project_root / "data" / "colorfull_background"
    token_path = project_root / "data" / "tokens" 
    dest_path = project_root / "data" / "images_generated_with_token"

    p = argparse.ArgumentParser(
        description="Générer des images 4000x2662 à partir de crops"
    )
    p.add_argument(
        "--src_dir",
        default=src_path,
        help="Dossier source des images à coller",
    )
    p.add_argument("--out_dir", default=dest_path, help="Dossier de sortie")
    p.add_argument("--count", type=int, default=3, help="Nombre d'images générées")
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
    p.add_argument(
        "--bg_dir",
        default=bg_path,
        help="Dossier des images de fond colorées (tirage 50/50 avec le fond uni)",
    )
    p.add_argument(
        "--grayscale",
        action="store_true",
        help="Convertir l'image finale en niveaux de gris",
    )
    p.add_argument(
        "--token_image",
        type=str,
        default=token_path,
        help="Chemin vers une image spéciale à inclure systématiquement",
    )
    args = p.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = load_source_images(src_dir)
    if not images:
        print(f"Aucune image trouvée dans {src_dir}.")
        return

    bg_images = load_source_images(Path(args.bg_dir))
    if not bg_images:
        print(f"Aucune image de fond trouvée dans {args.bg_dir}.")

    token_image_path = Path(args.token_image) if args.token_image else None

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
                bg_images=bg_images,
                make_grayscale=args.grayscale,
                token_image=token_image_path,
            )
            for r in records:
                filename = r[0]
                # If filtering is enabled and name matches pattern (e.g., 'A_image.png'), strip the prefix
                if args.grayscale and len(filename) >= 2 and filename[0].isalpha() and filename[1] == '_':
                    filename = filename[2:]
                
                writer.writerow([img_path.name, filename, r[1], r[2], r[3]])
            print(f"Wrote {img_path}")
    print(f"Placements CSV: {csv_path}")


if __name__ == "__main__":
    main()

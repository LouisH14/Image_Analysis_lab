#!/usr/bin/env python3
"""
Visualize model predictions vs ground truth on test images.
"""

import os
import sys
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.simple_detector import SimpleDetector


def load_model(checkpoint_path, device, num_classes=1, in_channels=4):
    model = SimpleDetector(num_classes=num_classes, in_channels=in_channels)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def visualize_predictions(test_dir, model, device, output_dir, num_samples=5):
    """Load test images, run predictions, and save visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    placements_csv = Path(test_dir) / "placements_1778601039.csv"
    if not placements_csv.exists():
        # Find any placements CSV in test_dir
        csvs = list(Path(test_dir).glob("placements*.csv"))
        if csvs:
            placements_csv = csvs[0]
        else:
            print(f"No placements CSV found in {test_dir}")
            return

    print(f"Using placements CSV: {placements_csv}")

    # Load placements
    placements = {}
    with open(placements_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["big_image"]
            placements[fname] = {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "angle": float(row["angle_deg"]),
            }

    image_files = list(Path(test_dir).glob("gen*.jpg"))[:num_samples]
    print(f"Processing {len(image_files)} test images...")

    for img_path in image_files:
        fname = img_path.name
        if fname not in placements:
            print(f"  No GT for {fname}, skipping")
            continue

        gt = placements[fname]

        # Load and preprocess image
        img = Image.open(img_path).convert("RGB")
        W_orig, H_orig = img.size

        # Resize to model input (512x512 as used in training)
        H_in, W_in = 512, 512
        img_resized = img.resize((W_in, H_in), Image.BILINEAR)

        # Preprocess: RGB + Saturation mask (Channel 4)
        img_np = np.array(img_resized)
        rgb_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        hsv_img = img_resized.convert("HSV")
        _, s, _ = hsv_img.split()
        s_mask = torch.from_numpy(np.array(s)).unsqueeze(0).float()
        hsv_mask = (s_mask > 100).float()

        x_tensor = torch.cat([rgb_tensor, hsv_mask], dim=0)
        x_tensor = x_tensor.unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            out = model(x_tensor)

        # out shape: (1, C, Hp, Wp) where C = 1 (obj) + 2 (offset) + 1 (angle) + num_classes
        _, _, Hp, Wp = out.shape
        out_np = out[0].cpu().numpy()  # (C, Hp, Wp)

        # Extract channels
        obj = 1 / (1 + np.exp(-out_np[0]))  # Sigmoid manuel
        offset_xy = out_np[2:4]  # dx, dy sont aux indices 2 et 3
        sin_cos = out_np[4:6]  # sin, cos sont aux indices 4 et 5

        # Find peak objectness
        y_pred, x_pred = np.unravel_index(np.argmax(obj), obj.shape)
        obj_conf = obj[y_pred, x_pred]
        offset_pred = offset_xy[:, y_pred, x_pred]
        sc_pred = sin_cos[:, y_pred, x_pred]
        angle_pred_rad = np.arctan2(sc_pred[0], sc_pred[1])
        angle_pred = np.rad2deg(angle_pred_rad)

        # Map from grid to image coordinates
        stride = W_in / Wp
        x_pred_img = (x_pred + offset_pred[0]) * stride
        y_pred_img = (y_pred + offset_pred[1]) * stride

        # Scale to original image
        scale_x = W_orig / W_in
        scale_y = H_orig / H_in
        x_pred_orig = x_pred_img * scale_x
        y_pred_orig = y_pred_img * scale_y

        # GT in original image coords (already at 4000x2662)
        x_gt = gt["x"]
        y_gt = gt["y"]
        angle_gt = gt["angle"]

        # Compute error
        pos_err = np.sqrt((x_pred_orig - x_gt) ** 2 + (y_pred_orig - y_gt) ** 2)
        angle_err = abs(angle_pred - angle_gt)

        print(f"{fname}:")
        print(f"  GT: ({x_gt:.1f}, {y_gt:.1f}), angle={angle_gt:.2f}°")
        print(
            f"  Pred: ({x_pred_orig:.1f}, {y_pred_orig:.1f}), angle={angle_pred:.2f}°, conf={obj_conf:.4f}"
        )
        print(f"  Errors: pos={pos_err:.1f}px, angle={angle_err:.2f}°")

        # Draw on original image
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)

        # GT: green circle
        r = 30
        draw.ellipse([x_gt - r, y_gt - r, x_gt + r, y_gt + r], outline="green", width=3)
        draw.text((x_gt, y_gt - 40), f"GT", fill="green")

        # Pred: red circle
        draw.ellipse(
            [x_pred_orig - r, y_pred_orig - r, x_pred_orig + r, y_pred_orig + r],
            outline="red",
            width=3,
        )
        draw.text((x_pred_orig, y_pred_orig - 40), f"Pred", fill="red")

        out_path = os.path.join(output_dir, f"vis_{fname}")
        draw_img.save(out_path)
        print(f"  Saved: {out_path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_dir", help="Test images directory")
    parser.add_argument("--model", default="models/detector_small.pth")
    parser.add_argument("--output", default="scripts/vis_output")
    parser.add_argument("--num", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cpu")
    model = load_model(args.model, device, num_classes=1, in_channels=4)
    visualize_predictions(args.test_dir, model, device, args.output, args.num)


if __name__ == "__main__":
    main()

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from datasets.synth_dataset import (
    ClassificationSynthDataset,
    COLOR_NAMES,
    RANK_NAMES,
    SPECIAL_NAMES,
    IGNORE_INDEX,
)
from models.simple_detector import HierarchicalDetector
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_crops", default="data/images_crop")
    parser.add_argument("--weights", default="models/detector_small.pth")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--useless_prob", type=float, default=0.35)
    parser.add_argument("--special_prob", type=float, default=0.1)
    parser.add_argument(
        "--save_debug_samples",
        action="store_true",
        help="Save a small set of debug images with GT and prediction",
    )
    parser.add_argument("--debug_dir", default="scripts/vis_output/debug_eval_samples")
    return parser.parse_args()


def load_model(weights_path, in_channels=4):
    model = HierarchicalDetector(in_channels=in_channels)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate(model, dataset):
    total = 0
    correct_card = 0
    useless_total = 0
    useless_correct = 0
    card_total = 0
    card_correct = 0
    color_total = 0
    color_correct = 0
    rank_total = 0
    rank_correct = 0
    special_total = 0
    special_correct = 0
    rank_cond_total = 0
    rank_cond_correct = 0
    confusion = np.zeros((2, 2), dtype=np.int64)
    summary_rows = []

    for i in range(len(dataset)):
        img, target = dataset[i]
        is_card = int(target["is_card"].item())
        color = int(target["color"].item())
        rank = int(target["rank"].item())
        special = int(target["special"].item())

        with torch.no_grad():
            outputs = model(img.unsqueeze(0))
            card_pred = int(outputs["card_logits"].argmax(dim=1).item())

        total += 1
        correct_card += int(card_pred == is_card)
        confusion[is_card, card_pred] += 1

        # Initialize predictions for debug block to avoid potential UnboundLocalError
        color_pred = IGNORE_INDEX
        rank_pred = IGNORE_INDEX
        special_pred = IGNORE_INDEX

        if is_card == 0:
            useless_total += 1
            useless_correct += int(card_pred == is_card)
        else:
            card_total += 1
            card_correct += int(card_pred == is_card)

            color_ok = False

            if color != IGNORE_INDEX:
                color_pred = int(outputs["color_logits"].argmax(dim=1).item())
                color_total += 1
                color_correct += int(color_pred == color)
                color_ok = color_pred == color
            else:
                color_pred = IGNORE_INDEX

            if rank != IGNORE_INDEX:
                rank_pred = int(outputs["rank_logits"].argmax(dim=1).item())
                rank_total += 1
                rank_correct += int(rank_pred == rank)

                # Conditional rank accuracy: only if card and color are correct
                if card_pred == 1 and color_ok:
                    rank_cond_total += 1
                    rank_cond_correct += int(rank_pred == rank)
            else:
                rank_pred = IGNORE_INDEX

            if special != IGNORE_INDEX:
                special_pred = int(outputs["special_logits"].argmax(dim=1).item())
                special_total += 1
                special_correct += int(special_pred == special)
            else:
                special_pred = IGNORE_INDEX

        # Save annotated debug image for inspection if requested
        if dataset.save_debug_samples:
            debug_dir = Path(dataset.debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)

            # Convert tensor to PIL (drop HSV mask channel if present)
            try:
                if img.shape[0] == 4:
                    pil_img = TF.to_pil_image(img[:3])
                else:
                    pil_img = TF.to_pil_image(img)
            except Exception:
                # fallback: clamp and convert
                arr = (img[:3].clamp(0, 1).numpy() * 255).astype("uint8")
                arr = np.transpose(arr, (1, 2, 0))
                pil_img = Image.fromarray(arr)

            draw = ImageDraw.Draw(pil_img)
            if is_card == 0:
                gt_name = dataset.useless_name
            elif special != IGNORE_INDEX:
                gt_name = SPECIAL_NAMES[special]
            else:
                gt_name = f"{COLOR_NAMES[color]}_{RANK_NAMES[rank]}"

            if card_pred == 0:
                pred_name = dataset.useless_name
            else:
                pred_parts = ["card"]
                if color_pred != IGNORE_INDEX:
                    pred_parts.append(f"color={COLOR_NAMES[color_pred]}")
                if rank_pred != IGNORE_INDEX:
                    pred_parts.append(f"rank={RANK_NAMES[rank_pred]}")
                if special_pred != IGNORE_INDEX:
                    pred_parts.append(f"special={SPECIAL_NAMES[special_pred]}")
                pred_name = " | ".join(pred_parts)

            ok = card_pred == is_card
            text = f"GT: {gt_name} | Pred: {pred_name} | {'OK' if ok else 'ERR'}"
            draw.rectangle([0, 0, pil_img.width, 22], fill=(0, 0, 0))
            draw.text((4, 4), text, fill=(255, 255, 255))

            fname = (
                debug_dir
                / f"eval_{i:04d}_gt{is_card}_pred{card_pred}_{'ok' if ok else 'err'}.jpg"
            )
            pil_img.save(fname)

            summary_rows.append(
                {
                    "idx": i,
                    "gt_is_card": is_card,
                    "gt_name": gt_name,
                    "pred_is_card": card_pred,
                    "pred_name": pred_name,
                    "correct": int(ok),
                    "file": str(fname),
                    "gt_color": color,
                    "pred_color": color_pred,
                    "gt_rank": rank,
                    "pred_rank": rank_pred,
                    "gt_special": special,
                    "pred_special": special_pred,
                }
            )
    accuracy = correct_card / total if total > 0 else 0.0
    useless_acc = useless_correct / useless_total if useless_total > 0 else 0.0
    card_acc = card_correct / card_total if card_total > 0 else 0.0
    color_acc = color_correct / color_total if color_total > 0 else 0.0
    rank_acc = rank_correct / rank_total if rank_total > 0 else 0.0
    rank_cond_acc = rank_cond_correct / rank_cond_total if rank_cond_total > 0 else 0.0
    special_acc = special_correct / special_total if special_total > 0 else 0.0

    print(f"--- Classification results on {len(dataset)} synthetic samples ---")
    print(f"Card-vs-useless accuracy: {accuracy*100:.1f}%")
    print(
        f"Useless accuracy: {useless_acc*100:.1f}% ({useless_correct}/{useless_total})"
    )
    print(f"Card accuracy: {card_acc*100:.1f}% ({card_correct}/{card_total})")
    print(f"Color accuracy: {color_acc*100:.1f}% ({color_correct}/{color_total})")
    print(f"Rank accuracy: {rank_acc*100:.1f}% ({rank_correct}/{rank_total})")
    print(
        f"Special accuracy: {special_acc*100:.1f}% ({special_correct}/{special_total})"
    )
    print(f"Conditional Rank accuracy (card & color correct): {rank_cond_acc*100:.1f}% ({rank_cond_correct}/{rank_cond_total})")
    print(f"Cards predicted as useless: {int(confusion[1, 0])}")
    print(f"Useless predicted as card: {int(confusion[0, 1])}")

    # Save summary CSV if debug samples were recorded
    try:
        import csv

        if hasattr(dataset, "save_debug_samples") and dataset.save_debug_samples:
            debug_dir = Path(dataset.debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            csv_path = debug_dir / "eval_summary.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "idx",
                        "gt_is_card",
                        "gt_name",
                        "pred_is_card",
                        "pred_name",
                        "correct",
                        "file",
                        "gt_color",
                        "pred_color",
                        "gt_rank",
                        "pred_rank",
                        "gt_special",
                        "pred_special",
                    ],
                )
                writer.writeheader()
                for r in summary_rows:
                    writer.writerow(r)
            print(f"Saved eval summary to {csv_path}")
    except Exception:
        pass

    return {
        "accuracy": accuracy,
        "card_accuracy": card_acc,
        "useless_accuracy": useless_acc,
        "color_accuracy": color_acc,
        "rank_accuracy": rank_acc,
        "rank_conditional_accuracy": rank_cond_acc,
        "special_accuracy": special_acc,
        "confusion": confusion,
    }


def main():
    args = parse_args()
    weights_path = Path(args.weights)

    if not weights_path.exists():
        print(f"Erreur : le modèle {weights_path} n'existe pas. Entraîne-le d'abord.")
        return

    model = load_model(weights_path)
    metadata_path = Path("models/detector_small_classes.json")
    class_names = None
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        class_names = metadata.get("class_names")

    dataset = ClassificationSynthDataset(
        src_crops_dir=args.src_crops,
        img_size=(args.img_size, args.img_size),
        epoch_size=args.samples,
        use_hsv_mask=True,
        useless_prob=args.useless_prob,
        special_prob=args.special_prob,
        save_debug_samples=args.save_debug_samples,
        debug_dir=args.debug_dir,
    )

    if class_names is not None:
        print(f"Loaded {len(class_names)} crop names + hierarchical metadata")

    evaluate(model, dataset)


if __name__ == "__main__":
    main()

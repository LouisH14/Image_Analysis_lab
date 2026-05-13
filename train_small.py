import argparse
import json
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from models.simple_detector import HierarchicalDetector, count_parameters
from datasets.synth_dataset import (
    ClassificationSynthDataset,
    IGNORE_INDEX,
    COLOR_NAMES,
    RANK_NAMES,
    SPECIAL_NAMES,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src_crops", default="data/images_crop")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epoch_size", type=int, default=500)
    p.add_argument("--useless_prob", type=float, default=0.35)
    p.add_argument("--special_prob", type=float, default=0.1)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--rank_label_smoothing", type=float, default=0.05)
    p.add_argument("--card_loss_weight", type=float, default=1.0)
    p.add_argument("--color_loss_weight", type=float, default=1.0)
    p.add_argument("--rank_loss_weight", type=float, default=6.0)
    p.add_argument("--special_loss_weight", type=float, default=1.0)
    p.add_argument("--phase2_card_loss_weight", type=float, default=0.5)
    p.add_argument("--phase2_color_loss_weight", type=float, default=0.7)
    p.add_argument("--phase2_rank_loss_weight", type=float, default=8.0)
    p.add_argument("--phase2_special_loss_weight", type=float, default=0.7)
    p.add_argument("--save_debug_samples", action="store_true")
    p.add_argument("--debug_dir", default="scripts/vis_output/debug_train_samples")
    return p.parse_args()


def masked_cross_entropy(logits, targets, ignore_index=IGNORE_INDEX, label_smoothing=0.0):
    mask = targets != ignore_index
    if mask.sum().item() == 0:
        return logits.new_tensor(0.0)
    return nn.functional.cross_entropy(
        logits[mask],
        targets[mask],
        label_smoothing=label_smoothing,
    )


def train():
    args = parse_args()

    ds = ClassificationSynthDataset(
        src_crops_dir=args.src_crops,
        img_size=(args.img_size, args.img_size),
        use_hsv_mask=True,
        epoch_size=args.epoch_size,
        useless_prob=args.useless_prob,
        special_prob=args.special_prob,
        save_debug_samples=args.save_debug_samples,
        debug_dir=args.debug_dir,
    )

    model = HierarchicalDetector(in_channels=4)
    print("Model params:", count_parameters(model))

    device = torch.device("cpu")
    model.to(device)

    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch_losses = []
    epoch_card_accs = []
    epoch_color_accs = []
    epoch_rank_accs = []
    epoch_special_accs = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()

        if epoch < args.warmup_epochs:
            w_card = args.card_loss_weight
            w_color = args.color_loss_weight
            w_rank = args.rank_loss_weight
            w_special = args.special_loss_weight
        else:
            w_card = args.phase2_card_loss_weight
            w_color = args.phase2_color_loss_weight
            w_rank = args.phase2_rank_loss_weight
            w_special = args.phase2_special_loss_weight

        running = 0.0
        correct_card = 0
        total_card = 0
        correct_color = 0
        total_color = 0
        correct_rank = 0
        total_rank = 0
        correct_special = 0
        total_special = 0
        for i, (img, target) in enumerate(dl):
            img = img.to(device)
            target = {k: v.to(device) for k, v in target.items()}
            outputs = model(img)

            loss_card = nn.functional.cross_entropy(
                outputs["card_logits"], target["is_card"]
            )
            loss_color = masked_cross_entropy(outputs["color_logits"], target["color"])
            loss_rank = masked_cross_entropy(
                outputs["rank_logits"],
                target["rank"],
                label_smoothing=args.rank_label_smoothing,
            )
            loss_special = masked_cross_entropy(
                outputs["special_logits"], target["special"]
            )
            loss = (
                w_card * loss_card
                + w_color * loss_color
                + w_rank * loss_rank
                + w_special * loss_special
            )

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            card_pred = outputs["card_logits"].argmax(dim=1)
            correct_card += int((card_pred == target["is_card"]).sum().item())
            total_card += int(target["is_card"].numel())

            color_mask = target["color"] != IGNORE_INDEX
            if color_mask.any():
                color_pred = outputs["color_logits"][color_mask].argmax(dim=1)
                correct_color += int(
                    (color_pred == target["color"][color_mask]).sum().item()
                )
                total_color += int(color_mask.sum().item())

            rank_mask = target["rank"] != IGNORE_INDEX
            if rank_mask.any():
                rank_pred = outputs["rank_logits"][rank_mask].argmax(dim=1)
                correct_rank += int(
                    (rank_pred == target["rank"][rank_mask]).sum().item()
                )
                total_rank += int(rank_mask.sum().item())

            special_mask = target["special"] != IGNORE_INDEX
            if special_mask.any():
                special_pred = outputs["special_logits"][special_mask].argmax(dim=1)
                correct_special += int(
                    (special_pred == target["special"][special_mask]).sum().item()
                )
                total_special += int(special_mask.sum().item())

            if (i + 1) % 10 == 0:
                avg_loss = running / (i + 1)
                avg_card_acc = correct_card / total_card if total_card > 0 else 0.0
                print(
                    f"Epoch {epoch} iter {i+1}/{len(dl)} loss {avg_loss:.4f} card_acc {avg_card_acc*100:.1f}%"
                )
        avg = running / len(dl)
        avg_card_acc = correct_card / total_card if total_card > 0 else 0.0
        avg_color_acc = correct_color / total_color if total_color > 0 else 0.0
        avg_rank_acc = correct_rank / total_rank if total_rank > 0 else 0.0
        avg_special_acc = correct_special / total_special if total_special > 0 else 0.0
        epoch_losses.append(avg)
        epoch_card_accs.append(avg_card_acc)
        epoch_color_accs.append(avg_color_acc)
        epoch_rank_accs.append(avg_rank_acc)
        epoch_special_accs.append(avg_special_acc)
        print(
            f"Epoch {epoch} finished, avg loss {avg:.4f}, card_acc {avg_card_acc*100:.1f}%, color_acc {avg_color_acc*100:.1f}%, rank_acc {avg_rank_acc*100:.1f}%, special_acc {avg_special_acc*100:.1f}%, weights(card,color,rank,special)=({w_card:.2f},{w_color:.2f},{w_rank:.2f},{w_special:.2f}), time {time.time()-t0:.1f}s"
        )
    # save model
    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "models/detector_small.pth")
    print("Saved model to models/detector_small.pth")

    metadata = {
        "class_names": ds.class_names,
        "useless_name": ds.useless_name,
        "useless_idx": ds.useless_idx,
        "color_names": COLOR_NAMES,
        "rank_names": RANK_NAMES,
        "special_names": SPECIAL_NAMES,
    }
    with open("models/detector_small_classes.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("Saved class metadata to models/detector_small_classes.json")

    # save epoch losses and plot
    Path("plots").mkdir(parents=True, exist_ok=True)
    np.save("models/train_losses.npy", np.array(epoch_losses))
    np.save("models/train_card_accs.npy", np.array(epoch_card_accs))
    np.save("models/train_color_accs.npy", np.array(epoch_color_accs))
    np.save("models/train_rank_accs.npy", np.array(epoch_rank_accs))
    np.save("models/train_special_accs.npy", np.array(epoch_special_accs))
    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Avg loss")
    plt.title("Training loss per epoch")
    plt.grid(True)
    plt.savefig("plots/training_loss.png", bbox_inches="tight")
    plt.close()
    print("Saved training loss plot to plots/training_loss.png")

    plt.figure()
    plt.plot(
        range(1, len(epoch_card_accs) + 1),
        np.array(epoch_card_accs) * 100.0,
        marker="o",
        label="card",
    )
    plt.plot(
        range(1, len(epoch_color_accs) + 1),
        np.array(epoch_color_accs) * 100.0,
        marker="o",
        label="color",
    )
    plt.plot(
        range(1, len(epoch_rank_accs) + 1),
        np.array(epoch_rank_accs) * 100.0,
        marker="o",
        label="rank",
    )
    plt.plot(
        range(1, len(epoch_special_accs) + 1),
        np.array(epoch_special_accs) * 100.0,
        marker="o",
        label="special",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training accuracy per head")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/training_accuracy.png", bbox_inches="tight")
    plt.close()
    print("Saved training accuracy plot to plots/training_accuracy.png")


if __name__ == "__main__":
    train()

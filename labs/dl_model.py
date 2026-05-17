"""
MobileNetV3 Small model for UNO card multi-label classification.

Trained from scratch (no pretrained weights — competition rule).
Training: BCEWithLogitsLoss over 54 card classes (multi-hot targets).
Inference: sigmoid + threshold → list of predicted card names per area.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import mobilenet_v3_small

from dl_dataset import CARD_CLASSES, NUM_CLASSES, get_area_crop


# ─── Model ───────────────────────────────────────────────────────────────────

class UNOCardClassifier(nn.Module):
    """
    MobileNetV3 Small trained from scratch for UNO card multi-label classification.

    Architecture: MobileNetV3-Small features → AdaptiveAvgPool → [576]
                  → Linear(256) → Hardswish → Dropout → Linear(54)
    No pretrained weights (competition constraint).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()
        # weights=None → random initialisation, no ImageNet weights
        backbone = mobilenet_v3_small(weights=None)

        in_features = backbone.classifier[0].in_features  # 576
        backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(256, num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ─── Training helpers ─────────────────────────────────────────────────────────

def _make_criterion(device: torch.device, pos_weight_value: float = 6.0) -> nn.BCEWithLogitsLoss:
    """BCE with positive-class upweighting to compensate for label sparsity."""
    pos_weight = torch.full((NUM_CLASSES,), pos_weight_value, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Returns (val_loss, exact_match_acc, avg_per_class_acc)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        all_preds.append((torch.sigmoid(logits) > threshold).float().cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    gts   = torch.cat(all_labels)

    exact_acc    = (preds == gts).all(dim=1).float().mean().item()
    per_class_acc = (preds == gts).float().mean(dim=0).mean().item()
    return total_loss / len(loader.dataset), exact_acc, per_class_acc


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 100,
    lr: float = 3e-3,
    warmup_epochs: int = 10,
    device: torch.device | None = None,
    save_path: str | Path = "uno_mobilenet.pth",
    threshold: float = 0.5,
) -> nn.Module:
    """
    Training from scratch: linear LR warmup then cosine annealing.

    Training from random init needs a higher peak LR than fine-tuning and a
    warmup phase so early gradients don't corrupt batch-norm statistics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Linear warmup for `warmup_epochs`, then cosine decay to lr/100
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = _make_criterion(device)

    best_val_loss = float("inf")
    save_path = Path(save_path)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, exact_acc, pc_acc = evaluate(model, val_loader, criterion, device, threshold)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | lr={current_lr:.2e} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"exact={exact_acc:.3f} | per_class={pc_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    print(f"\nBest model saved → {save_path}  (val_loss={best_val_loss:.4f})")

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


# ─── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_area(
    model: nn.Module,
    area: np.ndarray | Image.Image,
    transform,
    device: torch.device,
    threshold: float = 0.4,
) -> list[str]:
    """
    Predict which cards are visible in one area crop.

    Returns a list of '<card_name> (<prob>)' strings, or ['EMPTY'].
    """
    model.eval()
    if isinstance(area, np.ndarray):
        area = Image.fromarray(area)

    tensor = transform(area).unsqueeze(0).to(device)
    probs  = torch.sigmoid(model(tensor)).squeeze(0).cpu().numpy()

    hits = [(CARD_CLASSES[i], float(probs[i])) for i in np.where(probs > threshold)[0]]
    hits.sort(key=lambda x: -x[1])

    if not hits:
        return ["EMPTY"]
    return [f"{name} ({prob:.2f})" for name, prob in hits]


@torch.no_grad()
def predict_game_state(
    model: nn.Module,
    im_obj,                   # core.image instance
    transform,
    device: torch.device,
    threshold: float = 0.4,
) -> dict[str, list[str]]:
    """
    Predict cards for all 5 positions in a game image.

    Args:
        im_obj: core.image instance (provides .get() → numpy array)
    Returns:
        dict with keys 'Center', 'Player 1', ..., 'Player 4'
    """
    positions = ["Center", "Player 1", "Player 2", "Player 3", "Player 4"]
    img_array = im_obj.get()

    return {
        pos: predict_area(model, get_area_crop(img_array, i), transform, device, threshold)
        for i, pos in enumerate(positions)
    }


def load_model(checkpoint_path: str | Path, device: torch.device | None = None) -> nn.Module:
    """Load a saved UNOCardClassifier from disk."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNOCardClassifier()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

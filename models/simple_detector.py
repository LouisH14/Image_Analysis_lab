import torch
import torch.nn as nn


class SimpleDetector(nn.Module):
    """Compact CNN classifier for 55-way synthetic card recognition.

    The network keeps a small convolutional backbone and uses global average
    pooling followed by a linear classification head.
    """

    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        # small backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        logits = self.head(f)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class HierarchicalDetector(nn.Module):
    """Lightweight multi-head classifier for card/useless, color, rank, and special cards."""

    def __init__(self, in_channels: int = 4, hidden_channels: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.card_head = nn.Linear(hidden_channels, 2)
        self.color_head = nn.Linear(hidden_channels, 4)
        self.rank_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_channels, 13),
        )
        self.special_head = nn.Linear(hidden_channels, 2)

    def forward(self, x):
        features = self.pool(self.backbone(x)).flatten(1)
        return {
            "card_logits": self.card_head(features),
            "color_logits": self.color_head(features),
            "rank_logits": self.rank_head(features),
            "special_logits": self.special_head(features),
        }


if __name__ == "__main__":
    m = SimpleDetector(num_classes=54)
    print("Params:", count_parameters(m))

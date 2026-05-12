import torch
import torch.nn as nn


class SimpleDetector(nn.Module):
    """Small convolutional detector producing per-cell predictions.

    Outputs a tensor of shape (B, C_out, Hf, Wf) where C_out = 1 + num_classes + 2 + 2
    (objectness, class_logits, dx, dy, sin, cos)
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        # small backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        out_channels = 1 + num_classes + 2 + 2
        self.head = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        f = self.backbone(x)
        out = self.head(f)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    m = SimpleDetector(num_classes=54)
    print('Params:', count_parameters(m))

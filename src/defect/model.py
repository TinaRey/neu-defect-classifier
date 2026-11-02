import torch
from torch import nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),  # padding=1 keeps H and W unchanged
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),                                     # halves H and W
    )


class SimpleCNN(nn.Module):
    """4 x (Conv -> ReLU -> MaxPool) -> AdaptiveAvgPool -> Linear.

    Output is raw logits (no softmax): CrossEntropyLoss applies log-softmax itself,
    and the API applies softmax at inference time.
    """

    def __init__(self, num_classes: int = 6, in_channels: int = 1, widths: tuple[int, ...] = (16, 32, 64, 128)):
        super().__init__()
        chans = (in_channels, *widths)
        self.features = nn.Sequential(*[conv_block(a, b) for a, b in zip(chans[:-1], chans[1:])])
        # Averages each feature map down to 1x1 whatever the input size, so the
        # classifier needs no hard-coded flatten size and the model has few parameters.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(widths[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N, C, H, W)
        x = self.pool(self.features(x))  # (N, widths[-1], 1, 1)
        return self.classifier(torch.flatten(x, 1))  # (N, num_classes)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

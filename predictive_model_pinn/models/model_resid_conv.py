import torch
from torch import nn
from torch.nn import functional as F


class ResNetPhasePredictor(nn.Module):
    def __init__(
        self,
        input_res=64,
        base_channels=32,
        output_dim=16 * 16 * 4 * 2,  # sin + cos
    ):
        super().__init__()

        self.input_res = input_res

        # --- STEM ---
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )

        # --- RESIDUAL STAGES ---
        self.layer1 = self._make_layer(base_channels, 64, stride=2)  # 64 → 32
        self.layer2 = self._make_layer(64, 128, stride=2)  # 32 → 16
        self.layer3 = self._make_layer(128, 256, stride=2)  # 16 → 8
        # self.layer4 = self._make_layer(256, 256, stride=2)  # 8 → 4

        # --- GLOBAL HEAD ---
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, output_dim),
        )

    def _make_layer(self, in_ch, out_ch, stride):
        return nn.Sequential(
            ResidualBlock(in_ch, out_ch, stride),
            ResidualBlock(out_ch, out_ch, 1),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.pool(x)
        x = self.head(x)

        # --- SIN/COS → PHASE ---
        B = x.shape[0]

        outputs = x.view(B, 4, 16, 16, 2)

        sin, cos = torch.chunk(outputs, 2, dim=-1)

        phases = torch.atan2(sin, cos).squeeze(-1)

        return phases


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # match dimensions if needed
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.act(out)

        return out

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import BatchNorm2d


class ConvModel(nn.Module):
    def __init__(
        self, input_dim=64 * 64, output_dim=16 * 16 * 4, hidden_dim=4098
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3),
            nn.SiLU(inplace=True),
            nn.Conv2d(3, 9, kernel_size=3),
            nn.SiLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(9, 18, kernel_size=3),
            nn.SiLU(inplace=True),
            nn.Conv2d(18, 24, kernel_size=3),
            nn.SiLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(output_dim * 2),
            nn.SiLU(),
            nn.LazyLinear(output_dim * 2),
        )

    def forward(self, x) -> torch.Tensor:
        out = self.network(x)
        outputs_reshaped = out.view(-1, 4, 16, 16, 2)
        sin, cos = torch.chunk(outputs_reshaped, 2, dim=-1)
        phases = torch.atan2(sin, cos).squeeze(-1)
        return phases

    # def sample(self, x) -> torch.Tensor:
    #     mu, std = self.forward(x)
    #     print(mu, std)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std

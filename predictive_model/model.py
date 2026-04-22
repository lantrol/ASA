import torch
from torch import nn
from torch.nn import functional as F


class PredModel(nn.Module):
    def __init__(
        self, input_dim=64 * 64, output_dim=16 * 16 * 4 * 2, hidden_dim=2048
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x) -> torch.Tensor:
        return self.network(x)

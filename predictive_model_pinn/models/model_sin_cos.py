import torch
from torch import nn
from torch.nn import functional as F


class SinCosModel(nn.Module):
    def __init__(
        self, input_dim=64 * 64, output_dim=16 * 16 * 4, hidden_dim=4098
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim * 2),  # Predicts Sin and Cos
        )

    def forward(self, x) -> torch.Tensor:
        out = self.network(x)
        outputs_reshaped = out.view(-1, 4, 16, 16, 2)
        sin, cos = torch.chunk(outputs_reshaped, 2, dim=-1)
        phases = torch.atan2(sin, cos).squeeze(-1)
        return phases

    def get_phases(self, x) -> torch.Tensor:
        out = self.network(x)
        out_reshaped = out.view(-1, 4, 16, 16, 2)
        sin, cos = torch.chunk(out_reshaped, 2, dim=-1)
        phases = torch.atan2(sin, cos)
        return phases

    # def sample(self, x) -> torch.Tensor:
    #     mu, std = self.forward(x)
    #     print(mu, std)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std

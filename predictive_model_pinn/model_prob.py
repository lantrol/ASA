import torch
from torch import nn
from torch.nn import functional as F


class PredModel(nn.Module):
    def __init__(
        self, input_dim=64 * 64, output_dim=16 * 16 * 4, hidden_dim=4098
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim * 2 * 2),  # Predicts Sin and Cos
        )

    def forward(self, x):
        out = self.network(x)
        out_reshaped = out.view(-1, 4, 16 * 16 * 4)
        mu, logvar = torch.chunk(out_reshaped, 2, dim=-1)

        sin_mu, cos_mu = torch.chunk(mu, 2, dim=-1)
        sin_logvar, cos_logvar = torch.chunk(logvar, 2, dim=-1)

        sin = sin_mu + torch.exp(sin_logvar) * torch.randn(
            sin_mu.shape[-1], device="cuda"
        )
        cos = cos_mu + torch.exp(cos_logvar) * torch.randn(
            sin_mu.shape[-1], device="cuda"
        )

        phases = torch.atan2(sin, cos)
        return phases.view(-1, 4, 16, 16), mu, logvar

    def get_phases(self, x) -> torch.Tensor:
        out = self.network(x)
        out_reshaped = out.view(-1, 4, 16, 16, 4)
        sin_mu, sin_logvar, cos_mu, cos_logvar = torch.chunk(out_reshaped, 4, dim=-1)

        sin = sin_mu + torch.exp(sin_logvar) * torch.randn(1)
        cos = cos_mu + torch.exp(cos_logvar) * torch.randn(1)

        phases = torch.atan2(sin, cos)
        return phases

    # def sample(self, x) -> torch.Tensor:
    #     mu, std = self.forward(x)
    #     print(mu, std)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std

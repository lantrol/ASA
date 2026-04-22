import torch
from torch import nn
from torch.nn import functional as F


class MultiModel(nn.Module):
    def __init__(
        self, input_dim=64 * 64, output_dim=16 * 16, hidden_dim=2048 + 1024
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        self.activation = nn.SiLU(inplace=True)

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )

        self.l1 = nn.Linear(hidden_dim, output_dim * 2)
        self.l2 = nn.Linear(hidden_dim, output_dim * 2)
        self.l3 = nn.Linear(hidden_dim, output_dim * 2)
        self.l4 = nn.Linear(hidden_dim, output_dim * 2)

    def forward(self, x) -> torch.Tensor:
        x_1 = self.network(x)

        vals_1 = self.l1(x_1)
        vals_2 = self.l2(x_1)
        vals_3 = self.l3(x_1)
        vals_4 = self.l4(x_1)

        all = torch.cat((vals_1, vals_2, vals_3, vals_4), dim=-1)
        outputs_reshaped = all.view(-1, 4, 16, 16, 2)
        sin, cos = torch.chunk(outputs_reshaped, 2, dim=-1)
        phases = torch.atan2(sin, cos).squeeze(-1)

        return phases

    def forward_old(self, x) -> torch.Tensor:
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

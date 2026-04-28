import torch
from torch import nn
from torch.nn import functional as F


class ResidModel(nn.Module):
    def __init__(self, input_dim=64 * 64, output_dim=16 * 16, hidden_dim=4096) -> None:
        super().__init__()
        self.output_dim = output_dim

        self.activation = nn.SiLU(inplace=True)

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.l2 = nn.Linear(hidden_dim, output_dim * 2)
        self.ln2 = nn.LayerNorm(output_dim * 2)

        self.l3 = nn.Linear(output_dim * 2, output_dim * 2)
        self.ln3 = nn.LayerNorm(output_dim * 2)

        self.l4 = nn.Linear(output_dim * 2, output_dim * 2)
        self.ln4 = nn.LayerNorm(output_dim * 2)

        self.l5 = nn.Linear(output_dim * 2, output_dim * 2)
        self.ln5 = nn.LayerNorm(output_dim * 2)

        self.l6 = nn.Linear(output_dim * 2, output_dim * 2)
        self.ln6 = nn.LayerNorm(output_dim * 2)

    def forward(self, x) -> torch.Tensor:
        x_1 = self.ln1(self.activation(self.l1(x)))
        x_1 = self.ln2(self.activation(self.l2(x_1)))

        vals_1 = self.ln3(self.activation(self.l3(x_1))) + x_1
        vals_2 = self.ln4(self.activation(self.l4(vals_1))) + vals_1
        vals_3 = self.ln5(self.activation(self.l5(vals_2))) + vals_2
        vals_4 = self.ln6(self.activation(self.l6(vals_3))) + vals_3

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

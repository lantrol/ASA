import torch
import torch.nn as nn
import torch.nn.functional as F


class SinCosAttentionModel(nn.Module):
    def __init__(
        self,
        input_dim=64 * 64,
        hidden_dim=256,
        num_fields=4,
        field_size=16 * 16,
        num_heads=8,
        num_layers=4,
    ):
        super().__init__()

        self.num_fields = num_fields
        self.field_size = field_size

        # --- Image encoder ---
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )

        # --- Learnable field tokens ---
        self.field_tokens = nn.Parameter(torch.randn(num_fields, hidden_dim))

        # --- Transformer for field interaction ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Decoder: each token → 16x16 sin/cos ---
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, field_size * 2),
        )

    def forward(self, x):
        B = x.shape[0]

        # Encode image
        img_feat = self.encoder(x)  # (B, hidden_dim)

        # Expand field tokens per batch
        tokens = self.field_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, 4, hidden_dim)

        # Inject global image context
        tokens = tokens + img_feat.unsqueeze(1)

        # Self-attention across the 4 fields
        tokens = self.transformer(tokens)  # (B, 4, hidden_dim)

        # Decode each field independently AFTER interaction
        out = self.decoder(tokens)  # (B, 4, 16*16*2)

        # Reshape
        out = out.view(B, self.num_fields, 16, 16, 2)

        sin, cos = torch.chunk(out, 2, dim=-1)
        phases = torch.atan2(sin, cos).squeeze(-1)

        return phases

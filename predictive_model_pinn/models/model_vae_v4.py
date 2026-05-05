import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding2D(nn.Module):
    def __init__(self, height, width, dim):

        super().__init__()

        self.pos_embed = nn.Parameter(torch.zeros(1, height, width, dim))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):

        return x + self.pos_embed


class PatchVAETransformer(nn.Module):
    def __init__(
        self,
        input_res=64,
        patch_size=8,
        latent_dim=256,
        transformer_depth=2,
        num_heads=8,
        ff_dim=512,
        output_dim=16 * 16 * 4 * 2,
    ):

        super().__init__()

        assert input_res % patch_size == 0, "input_res must be divisible by patch_size"

        self.input_res = input_res

        self.patch_size = patch_size

        self.grid_size = input_res // patch_size

        self.latent_dim = latent_dim

        # --- 1. PATCH EMBEDDING (FIXED) ---

        # This cleanly creates patch tokens

        self.patch_embed = nn.Sequential(
            # --- Stage 1: low-level features ---
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            # --- Stage 2: mild downsampling ---
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 → 32
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            # --- Stage 3: project to latent_dim + patching ---
            nn.Conv2d(
                64,
                self.latent_dim,
                kernel_size=self.patch_size // 2,  # softer than full patch
                stride=self.patch_size // 2,
            ),
        )

        # --- 2. VAE HEADS ---

        self.fc_mu = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)

        self.fc_logvar = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)

        # --- 3. POSITIONAL ENCODING ---

        self.pos_encoding = PositionalEncoding2D(
            self.grid_size, self.grid_size, latent_dim
        )

        # --- 4. TRANSFORMER ---

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation="gelu",  # safer than 'silu' here
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_depth
        )

        # --- 5. DECODER ---

        # Upscale from patch grid back to image

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, 2, 1),  # g → 2g
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 2g → 4g
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 4g → 8g
            nn.SiLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),  # 8g → 16g (64 if g=4)
            nn.Sigmoid(),
        )

        # --- 6. PHASE HEAD ---

        self.phase_head = nn.Linear(
            self.grid_size * self.grid_size * latent_dim, output_dim
        )

    def encode(self, x):

        x = self.patch_embed(x)  # [B, C, G, G]

        mu = self.fc_mu(x)

        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x, training=False):

        if x.dim() == 3:
            x = x.unsqueeze(1)

        # --- ENCODE ---

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        # --- TRANSFORMER ---

        z_tokens = z.permute(0, 2, 3, 1)  # [B, G, G, C]

        z_tokens = self.pos_encoding(z_tokens)

        B, G, _, C = z_tokens.shape

        z_seq = z_tokens.reshape(B, G * G, C)

        z_trans = self.transformer(z_seq)

        # --- PHASE HEAD ---

        phase_logits = self.phase_head(z_trans.reshape(B, -1))

        outputs = phase_logits.view(B, 4, 16, 16, 2)

        sin, cos = torch.chunk(outputs, 2, dim=-1)

        phases = torch.atan2(sin, cos).squeeze(-1)

        if training:
            # reshape back to spatial grid

            z_spatial = z_trans.reshape(B, G, G, C).permute(0, 3, 1, 2)

            # reconstruction = self.decoder(z_spatial)

            # print(f"Decoder: {reconstruction.shape}")

            return phases, mu, logvar

        return phases

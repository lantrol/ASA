import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, input_channels=1, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 128, 8, 8)
        result = self.decoder(result)
        return result

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_output))
        return x

class SinCosModel(nn.Module):
    def __init__(
        self, 
        input_res=64, 
        latent_dim=256, 
        transformer_depth=2, 
        num_heads=8, 
        ff_dim=512,
        output_dim=16 * 16 * 4 * 2 # 4 fields * 16*16 * sin/cos
    ) -> None:
        super().__init__()
        self.input_res = input_res
        self.latent_dim = latent_dim
        
        # VAE
        self.vae = VAE(input_channels=1, latent_dim=latent_dim)
        
        # Transformer
        self.transformer_layers = nn.Sequential(
            *[TransformerBlock(latent_dim, num_heads, ff_dim) for _ in range(transformer_depth)]
        )
        self.transformer_norm = nn.LayerNorm(latent_dim)
        
        # Decoder/Linear head
        # We want 4 * 16 * 16 * 2 parameters
        self.final_linear = nn.Linear(latent_dim, output_dim)

    def forward(self, x) -> torch.Tensor:
        # x shape: [B, 1, 64, 64]
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # VAE encoding
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        
        # Transformer (expects [B, Seq, Dim], treat latent as sequence of length 1 or split it)
        # To apply attention effectively, let's treat the latent as a sequence of tokens
        # For simplicity, we'll just treat the single latent vector as a sequence of length 1
        # but if latent_dim is large, we could reshape it.
        z_seq = z.unsqueeze(1) # [B, 1, latent_dim]
        z_trans = self.transformer_layers(z_seq)
        z_trans = self.transformer_norm(z_trans.squeeze(1)) # [B, latent_dim]
        
        # Final projection
        out = self.final_linear(z_trans)
        
        # Reshape to [B, 4, 16, 16, 2]
        outputs_reshaped = out.view(-1, 4, 16, 16, 2)
        sin, cos = torch.chunk(outputs_reshaped, 2, dim=-1)
        # Note: The chunking logic in original code was: sin, cos = torch.chunk(outputs_reshaped, 2, dim=-1)
        # But outputs_reshaped has dim -1 as 2. Chunking 2 into 2 gives two tensors of size 1.
        # Wait, the original code: outputs_reshaped = out.view(-1, 4, 16, 16, 2)
        # sin, cos = torch.chunk(outputs_reshaped, 2, dim=-1) 
        # This means sin is [B, 4, 16, 16, 1] and cos is [B, 4, 16, 16, 1]
        
        phases = torch.atan2(sin, cos).squeeze(-1)
        return phases

    def get_phases(self, x) -> torch.Tensor:
        # Returns sin/cos components as per original model
        if x.dim() == 3:
            x = x.unsqueeze(1)
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        z_seq = z.unsqueeze(1)
        z_trans = self.transformer_layers(z_seq)
        z_trans = self.transformer_norm(z_trans.squeeze(1))
        out = self.final_linear(z_trans)
        out_reshaped = out.view(-1, 4, 16, 16, 2)
        sin, cos = torch.chunk(out_reshaped, 2, dim=-1)
        phases = torch.atan2(sin, cos) # [B, 4, 16, 16, 1]
        return phases

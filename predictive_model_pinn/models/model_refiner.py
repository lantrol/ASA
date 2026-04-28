import torch
import torch.nn as nn


class RefinerSinCosModel(nn.Module):
    def __init__(self, input_dim=64 * 64, hidden_dim=4096):
        super().__init__()

        # Stage 1: Initial Prediction (The "Rough Guess")
        # Predicts 4 fields * 16 * 16 * 2 (sin/cos)
        self.initial_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4 * 16 * 16 * 2),
        )

        # Stage 2: The Refiner (Spatial & Inter-field interaction)
        # Input: 4 fields * 2 (sin/cos) = 8 channels
        # Output: 4 fields * 2 (sin/cos) = 8 channels
        self.refiner = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),  # Residual connection style
        )

    def forward(self, x) -> torch.Tensor:
        batch_size = x.shape[0]

        # 1. Initial MLP prediction
        out = self.initial_mlp(x)
        # Reshape to (Batch, Channels, H, W) where Channels = 4 fields * 2 (sin/cos)
        out = out.view(batch_size, 8, 16, 16)

        # 2. Refinement
        # The CNN looks at all 8 channels simultaneously.
        # This allows Field 1 to "see" Field 2 via the convolution kernels.
        refined = out + self.refiner(
            out
        )  # Residual connection helps training stability

        # 3. Reshape back to (B, 4, 16, 16, 2) for phase calculation
        # We need to be careful with the dimension ordering
        refined = refined.view(batch_size, 4, 2, 16, 16)
        refined = refined.permute(0, 1, 3, 4, 2)  # (B, 4, 16, 16, 2)

        sin = refined[..., 0]
        cos = refined[..., 1]

        phases = torch.atan2(sin, cos)
        return phases

import torch
import torch.nn as nn

from se3_world_model.layers import VNLinear, VNResBlock


class SE3Encoder(nn.Module):
    """
    Encodes Point Cloud [B, N, 3] -> Global Latent State [B, Latent, 3].
    Uses Point-wise VN layers followed by Mean Pooling.
    """
    def __init__(self, in_channels: int, latent_dim: int) -> None:
        super().__init__()
        # Lift raw points (1 channel) to high-dim vector features
        self.lift = VNLinear(in_channels, latent_dim)

        self.backbone = nn.Sequential(
            VNResBlock(latent_dim),
            VNResBlock(latent_dim)
        )

        self.pre_pool = VNLinear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, 3]
        B, N, _ = x.shape

        # Flatten B and N to treat points as independent samples initially
        # Input becomes [B*N, 1, 3] -> 1 feature per point (its position)
        x_flat = x.view(-1, 1, 3)

        # Feature Extraction
        feat: torch.Tensor = self.lift(x_flat)  # [B*N, Latent, 3]
        feat = self.backbone(feat)
        feat = self.pre_pool(feat)

        # Global Pooling (Mean is equivariant)
        # Reshape back to [B, N, Latent, 3]
        feat = feat.view(B, N, -1, 3)
        latent_global = feat.mean(dim=1)  # [B, Latent, 3]

        return latent_global


class SE3Decoder(nn.Module):
    """
    Decodes Global Latent State [B, Latent, 3] -> Point Cloud [B, N, 3].
    """
    def __init__(self, latent_dim: int, num_points: int) -> None:
        super().__init__()
        self.num_points = num_points

        self.net = nn.Sequential(
            VNResBlock(latent_dim),
            VNResBlock(latent_dim),
            # Map Latent Channels -> Number of Points
            VNLinear(latent_dim, num_points)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, Latent, 3]
        # Output: [B, Num_Points, 3]
        out: torch.Tensor = self.net(z)
        return out

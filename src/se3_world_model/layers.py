import torch
import torch.nn as nn

def safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Computes norm safely to avoid NaN gradients."""
    return torch.norm(x, dim=dim, keepdim=True).clamp(min=eps)

class VNLinear(nn.Module):
    """
    Vector Neuron Linear Layer.
    Maps vector features [B, C_in, 3] to [B, C_out, 3] equivariantly.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: [B, C, 3]
        # Transpose to [B, 3, C] because nn.Linear operates on the last dim
        x_transpose = x.transpose(1, 2)
        out = self.map_to_feat(x_transpose)
        return out.transpose(1, 2)

class VNLeakyReLU(nn.Module):
    """
    Vector Neuron LeakyReLU.
    Uses a learnable direction to split the vector space.
    """
    def __init__(self, in_channels: int, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, 3]
        # Learn direction k
        k = self.map_to_dir(x.transpose(1, 2)).transpose(1, 2)
        k_norm = safe_norm(k)
        
        # Projection: q = (x . k / |k|^2) * k
        dot_prod = (x * k).sum(dim=-1, keepdim=True)
        q = (dot_prod / (k_norm**2 + 1e-6)) * k
        
        # Orthogonal component: u = x - q
        u = x - q
        
        # Directional Masking
        mask = (dot_prod >= 0).float()
        
        # Leaky Logic: Scale parallel component if opposing direction
        return mask * x + (1 - mask) * (self.negative_slope * q + u)

class VNResBlock(nn.Module):
    """
    SE(3) Equivariant Residual Block.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            VNLinear(channels, channels),
            VNLeakyReLU(channels),
            VNLinear(channels, channels),
            VNLeakyReLU(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Explicit type hint for strict mode
        out: torch.Tensor = self.block(x)
        return x + out

class VNInvariant(nn.Module):
    """
    Extracts rotation-invariant scalars from vector features.
    Input: [B, C, 3] -> Output: [B, C_out]
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple invariant: norm of vectors
        inv = torch.norm(x, dim=-1) # [B, C]
        out: torch.Tensor = self.mlp(inv)
        return out

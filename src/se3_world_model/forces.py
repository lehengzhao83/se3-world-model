import torch
import torch.nn as nn


class ContextualForceGenerator(nn.Module):
    """
    Implicit Symmetry Breaking Adapter.
    Learns to generate 'correction forces' from non-equivariant context
    (e.g., Absolute Position, Time, Object ID).
    """
    def __init__(self, context_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        # Standard MLP breaks SE(3) structure intentionally
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Outputs a 3D vector
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [B, K] Scalar context features.
        Returns:
            [B, 1, 3] Force vector.
        """
        f: torch.Tensor = self.net(context)
        return f.unsqueeze(1)


def inject_global_vectors(
    features: torch.Tensor,
    vectors: torch.Tensor
) -> torch.Tensor:
    """
    Explicit Symmetry Breaking Injection.
    Concatenates known global vectors (Gravity, Wind) as Type-1 features.

    Args:
        features: [B, C, 3]
        vectors:  [B, N, 3]
    Returns:
        [B, C + N, 3]
    """
    if vectors.shape[0] != features.shape[0]:
        vectors = vectors.expand(features.shape[0], -1, -1)
    return torch.cat([features, vectors], dim=1)

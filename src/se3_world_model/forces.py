import torch
import torch.nn as nn
from se3_world_model.layers import safe_norm

class EquivariantContextModulator(nn.Module):
    """
    等变上下文调制器 (Equivariant Context Modulator)。
    替代原有的非等变 Force Generator。
    
    它接收所有的标量上下文（风力大小、物体质量、时间步等），输出通道权重，
    去动态缩放（Gating）等变向量特征。
    标量乘向量的操作严格保留了 SE(3) 等变性。
    """
    def __init__(self, context_dim: int, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        # 输出的维度是 latent_dim，即为每一个等变特征通道预测一个标量权重
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, context: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [B, context_dim] 标量上下文
            z: [B, latent_dim, 3] 需要被调制的等变特征
        """
        # weight: [B, latent_dim]
        weight = self.mlp(context)
        # 增加空间维度以进行广播相乘: [B, latent_dim, 1]
        weight = weight.unsqueeze(-1)
        
        # 标量调制：[B, latent_dim, 3] * [B, latent_dim, 1]
        return z * weight


def inject_global_vectors(
    features: torch.Tensor,
    vectors: torch.Tensor
) -> torch.Tensor:
    """
    显式等变向量注入器。
    将归一化后的方向向量（Type-1）拼接到隐特征中。
    """
    if vectors.ndim == 2:
        vectors = vectors.unsqueeze(1)
    if vectors.shape[0] != features.shape[0]:
        vectors = vectors.expand(features.shape[0], -1, -1)
        
    return torch.cat([features, vectors], dim=1)

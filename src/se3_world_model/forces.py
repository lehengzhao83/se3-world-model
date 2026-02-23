import torch
import torch.nn as nn


class ContextualForceGenerator(nn.Module):
    """
    隐式对称性破缺适配器 (Implicit Symmetry Breaking Adapter)。
    用于从非等变的上下文信息（例如绝对坐标、时间步、物体ID、标量风力等）中
    学习并生成物理空间中的“修正力”（Correction Forces）。
    """
    def __init__(self, context_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        # 这是一个标准的多层感知机 (MLP)。
        # 注意：这里故意使用了标准 MLP，因为它的操作不具备 SE(3) 等变性。
        # 目的是刻意“打破”系统原有的严格对称性（比如风从某个特定方向吹来，旋转对称性就不复存在了）。
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 最终输出一个 3D 的力向量 (fx, fy, fz)
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        前向传播：
        Args:
            context: 标量上下文特征张量。
        Returns:
            生成的修正力向量张量。
        """
        # 通过非等变 MLP 网络计算力向量
        f: torch.Tensor = self.net(context)
        # 增加一个维度以适配后续的等变张量拼接操作 (变成类似 [Batch, 1个通道, 3D向量] 的结构)
        return f.unsqueeze(1)


def inject_global_vectors(
    features: torch.Tensor,
    vectors: torch.Tensor
) -> torch.Tensor:
    """
    显式全局向量注入器 (Explicit Symmetry Breaking Injection)。
    将已知的、具有物理意义的全局向量（如重力加速度、整体环境风力等）
    作为 Type-1 特征（即标准的 3D 向量）直接与现有的等变隐特征进行拼接。

    Args:
        features: 当前模型的全局隐变量特征
        vectors:  需要注入的全局显式向量特征
    Returns:
        拼接后的特征张量
    """
    # 检查全局向量的 Batch 维度是否与隐特征的 Batch 维度一致
    if vectors.shape[0] != features.shape[0]:
        # 如果不一致（例如全局向量对整个 batch 是固定的），则在 Batch 维度上进行扩展（复制）
        vectors = vectors.expand(features.shape[0], -1, -1)
        
    # 在通道维度 (dim=1) 上将特征和全局向量拼接起来，使得网络在演化时可以感知到这些全局物理量
    return torch.cat([features, vectors], dim=1)

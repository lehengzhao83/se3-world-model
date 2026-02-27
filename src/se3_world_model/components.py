import torch
import torch.nn as nn

# 引入重构后的新算子
from se3_world_model.layers import VNLinear, VNGatedBlock, EquivariantTemporalAttention

class SE3Encoder(nn.Module):
    """
    重构后的 SE(3) 等变编码器 (SE3 Equivariant Temporal Encoder)
    
    输入维度：
    [B, N, H, C_in, 3] 
      - B: Batch size
      - N: Number of points (点数)
      - H: History Length (历史时间步数，对应序列长度)
      - C_in: Input Channels (当前为3：局部位置、速度、接触力)
      - 3: 3D 向量 (x, y, z)
    """
    def __init__(self, in_channels: int, latent_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        
        # 1. 特征提升层 (逐时间步、逐点独立映射)
        self.lift = VNLinear(in_channels, latent_dim)

        # 2. 等变主干网络 (替换为带有跨通道信息交互的 VNGatedBlock)
        # 这里使得不同物理通道（如速度与力）可以在不破坏等变性的前提下产生非线性反应
        self.backbone = nn.Sequential(
            VNGatedBlock(latent_dim),
            VNGatedBlock(latent_dim)
        )

        # 3. 沿时间轴的等变自注意力机制 (Temporal Attention)
        # 聚合 H 维度的历史信息，提取关键碰撞或变轨特征
        self.temporal_attn = EquivariantTemporalAttention(channels=latent_dim, num_heads=num_heads)

        # 4. 池化前预处理层
        self.pre_pool = VNLinear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: [B, N, H, C_in, 3]
        B, N, H, C_in, _ = x.shape
        
        # === 步骤 1: 逐点、逐时间步特征提取 ===
        # 将 Batch, Points, History 融合到一起，交由 Vector Neurons 处理
        x_flat = x.reshape(B * N * H, C_in, 3)
        
        # 提升维度 -> Gated 非线性特征交互
        feat = self.lift(x_flat)
        feat = self.backbone(feat)
        
        # === 步骤 2: 时间维度的等变注意力聚合 (Temporal Aggregation) ===
        # 恢复出 Sequence (H) 维度：[B*N, H, Latent, 3]
        feat_seq = feat.reshape(B * N, H, -1, 3)
        
        # 使用内积 Attention 在时间轴上聚合特征
        # 输出维度依然是 [B*N, H, Latent, 3]
        attn_out = self.temporal_attn(feat_seq)
        
        # 将聚合后的时间序列压平 (可以选择取最后一个时间步，或者对 H 取均值)
        # 这里我们取包含全局历史注意力的最后一个时间步特征
        feat_t_agg = attn_out[:, -1, :, :] # [B*N, Latent, 3]

        # === 步骤 3: 全局池化 (Spatial Global Pooling) ===
        feat_t_agg = self.pre_pool(feat_t_agg)
        
        # 恢复 (B, N) 空间维度: [B, N, Latent, 3]
        # 空间 Mean Pooling 聚合所有点
        feat_spatial = feat_t_agg.reshape(B, N, -1, 3)
        
        # 空间 Mean Pooling 聚合所有点，维持严格的 SE(3) 等变性
        # 结果维度: [B, Latent, 3]
        latent_global = feat_spatial.mean(dim=1)

        return latent_global

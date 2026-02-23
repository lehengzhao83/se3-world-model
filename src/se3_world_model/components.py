import torch
import torch.nn as nn

from se3_world_model.layers import VNLinear, VNResBlock

class SE3Encoder(nn.Module):
    """
    SE(3) 等变编码器 (SE3 Equivariant Encoder)
    
    功能：
    将输入的点云序列（包含位置和速度信息）编码为一个全局的、保持旋转等变性的隐状态 z。
    
    输入维度：
    [B, N, C, 3]
      - B: Batch size
      - N: Number of points (点数)
      - C: Input Channels (输入通道数，通常为 2*History_Len，代表位置+速度的历史)
      - 3: 3D 向量 (x, y, z)
      
    输出维度：
    [B, Latent_Dim, 3]
      - 聚合了所有点信息的全局特征，每个通道都是一个 3D 向量，随刚体旋转而旋转。
    """
    def __init__(self, in_channels: int, latent_dim: int) -> None:
        super().__init__()
        
        # 1. 特征提升层 (Lifting Layer)
        # 将输入的原始物理向量（位置/速度通道）映射到高维的等变特征空间。
        # 输入通道: in_channels -> 输出通道: latent_dim
        self.lift = VNLinear(in_channels, latent_dim)

        # 2. 等变主干网络 (Equivariant Backbone)
        # 在高维特征空间进行非线性处理，提取更复杂的几何特征。
        # 使用 VNResBlock 保证深层网络的梯度传播和特征表达能力。
        self.backbone = nn.Sequential(
            VNResBlock(latent_dim),
            VNResBlock(latent_dim)
        )

        # 3. 池化前预处理层 (Pre-pooling Layer)
        # 在聚合所有点的信息之前，对特征进行进一步的混合。
        self.pre_pool = VNLinear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 可能有两种形状:
        # 1. [B, N, 3]    -> 仅有当前位置 (C=1 被省略)
        # 2. [B, N, C, 3] -> 包含历史信息或速度 (C > 1)
        
        # === 步骤 1: 数据展平 (Flattening) ===
        # Vector Neurons (VN) 层通常处理形状为 [Batch, Channels, 3] 的数据。
        # 这里我们需要把 "Batch" 和 "Points" 维度合并，把每个点视为一个独立的样本进行并行处理。
        
        if x.ndim == 3:
            # 情况 1: 输入只有单通道 [B, N, 3]
            B, N, _ = x.shape
            # 变形为 [B*N, 1, 3] 以适配 VNLinear
            x_flat = x.view(-1, 1, 3)
        else:
            # 情况 2: 输入有多通道 [B, N, C, 3]
            B, N, _, _ = x.shape
            # 变形为 [B*N, C, 3]，保持最后 3 维向量结构不变
            x_flat = x.view(B * N, -1, 3)

        # === 步骤 2: 逐点特征提取 (Point-wise Feature Extraction) ===
        # 以下操作都是"局部"的，即每个点独立计算，互不干扰。
        
        # 提升维度: [B*N, C, 3] -> [B*N, Latent, 3]
        feat: torch.Tensor = self.lift(x_flat)
        
        # 深层特征提取
        feat = self.backbone(feat)
        feat = self.pre_pool(feat)

        # === 步骤 3: 全局池化 (Global Pooling) ===
        # 将所有点的信息聚合为一个全局向量。
        
        # 先恢复出 (B, N) 维度: [B, N, Latent, 3]
        feat = feat.view(B, N, -1, 3)
        
        # 使用 Mean Pooling (均值池化)。
        # 注意：均值操作是线性的，因此完美的保留了旋转等变性 (Equivariance)。
        # 如果输入整体旋转了 R，那么所有点的特征旋转 R，均值自然也旋转 R。
        # 结果维度: [B, Latent, 3]
        latent_global = feat.mean(dim=1)

        return latent_global


class SE3Decoder(nn.Module):
    """
    SE(3) 等变解码器 (SE3 Equivariant Decoder)
    
    功能：
    （可选组件）将全局隐状态 z 解码回点云空间。
    注意：在 model.py 的主模型 SE3WorldModel 中，实际上使用的是 SE3RigidDecoder（刚体解码器），
    这个类主要用于重构任务或调试，将隐变量还原为 N 个点的坐标。
    
    输入：[B, Latent_Dim, 3]
    输出：[B, Num_Points, 3]
    """
    def __init__(self, latent_dim: int, num_points: int) -> None:
        super().__init__()
        self.num_points = num_points

        self.net = nn.Sequential(
            VNResBlock(latent_dim),
            VNResBlock(latent_dim),
            # 这里的 VNLinear 充当了"生成器"的角色：
            # 它将特征通道数 (Latent_Dim) 映射为点数 (Num_Points)。
            # 也就是说，输出的每个"通道"实际上代表一个点的坐标向量。
            VNLinear(latent_dim, num_points)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # 输入 z: [B, Latent, 3]
        
        # 解码生成点云: [B, Num_Points, 3]
        out: torch.Tensor = self.net(z)
        return out

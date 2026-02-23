import torch
import torch.nn as nn


def safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """
    安全的向量求模（L2 Norm）函数。
    为防止在向量长度趋近于 0 时，求导产生 NaN (梯度爆炸/消失) 错误，
    使用 clamp 强制将模长的最小值限制为 eps (如 1e-6)。
    """
    return torch.norm(x, dim=dim, keepdim=True).clamp(min=eps)


class VNLinear(nn.Module):
    """
    向量神经元线性层 (Vector Neuron Linear Layer)。
    将输入向量特征进行等变线性映射。
    它的核心思想是：在通道维度上进行线性组合，而不改变最后的 3D 向量空间结构。
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # 传统的 nn.Linear。注意 bias=False 是强制要求的，
        # 因为直接加上一个常数偏置向量会打破平移/旋转的等变性。
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入张量 x 的格式是：[B, 通道数C, 3D坐标]
        
        # 因为 PyTorch 的 nn.Linear 默认是作用在张量的最后一个维度上，
        # 为了让线性变换作用于“通道(C)”而不是“3D坐标(3)”，必须先转置。
        # 转置后: [B, 3, C]
        x_transpose = x.transpose(1, 2)
        
        # 在通道维度进行特征映射: [B, 3, C_in] -> [B, 3, C_out]
        out = self.map_to_feat(x_transpose)
        
        # 将结果重新转置回标准的 Vector Neuron 格式: [B, C_out, 3]
        return out.transpose(1, 2)


class VNLeakyReLU(nn.Module):
    """
    向量神经元 LeakyReLU 激活函数 (Vector Neuron LeakyReLU)。
    普通 ReLU 无法直接作用于 3D 向量（会破坏旋转等变性）。
    VNLeakyReLU 的做法是：网络自己学习一个方向向量 k，将输入向量投影到 k 上，
    根据投影方向（同向或反向）来决定是否施加衰减系数 (negative_slope)。
    """
    def __init__(self, in_channels: int, negative_slope: float = 0.2) -> None:
        super().__init__()
        # 用于学习判别方向 k 的线性层，同样不能有 bias
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope # 负半轴的缩放比例

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 学习判别方向矩阵 k
        # 使用线性层学习出与输入同维度的方向张量，并转置回 [B, C, 3]
        k = self.map_to_dir(x.transpose(1, 2)).transpose(1, 2)
        # 求出方向向量的模长（使用前面定义的 safe_norm 防除零）
        k_norm = safe_norm(k)

        # 2. 将输入 x 投影到方向 k 上
        # 求 x 和 k 的内积 (点乘)
        dot_prod = (x * k).sum(dim=-1, keepdim=True)
        # 得到投影后的平行分量 q: q = (x·k / |k|^2) * k
        q = (dot_prod / (k_norm**2 + 1e-6)) * k

        # 3. 计算正交分量 u
        # u 是去除了平行分量后的垂直部分 (u 垂直于 k)
        u = x - q

        # 4. 根据点积结果生成掩码 (Mask)
        # 如果点积 >= 0 (即夹角 <= 90度)，mask 为 1，表示处于正半区
        # 否则 mask 为 0，表示处于负半区
        mask = (dot_prod >= 0).float()

        # 5. Leaky 逻辑应用
        # 正半区 (mask=1): 保持原向量 x 不变
        # 负半区 (mask=0): 仅对平行于 k 的分量 q 进行缩放(乘以负半轴斜率)，加上正交分量 u
        return mask * x + (1 - mask) * (self.negative_slope * q + u)


class VNResBlock(nn.Module):
    """
    SE(3) 等变残差块 (Equivariant Residual Block)。
    组合 Vector Neuron 层，构建深层网络，并引入残差连接解决梯度消失问题。
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        # 标准的线性 -> 激活 -> 线性 -> 激活的堆叠结构，全替换为 VN 版本
        self.block = nn.Sequential(
            VNLinear(channels, channels),
            VNLeakyReLU(channels),
            VNLinear(channels, channels),
            VNLeakyReLU(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 类型提示帮助静态分析工具
        out: torch.Tensor = self.block(x)
        # 残差连接：将输入直接加到输出上
        return x + out


class VNInvariant(nn.Module):
    """
    旋转不变性特征提取器 (Rotation-invariant Feature Extractor)。
    将 3D 的等变向量特征，降维/提取为完全不受旋转影响的“标量”特征 (Scalars)。
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # 因为输入已经是标量（向量的模长），这里可以使用标准的多层感知机 (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 提取标量不变量：直接求每个通道中 3D 向量的模长（长度）
        # 无论坐标系如何旋转，向量的长度永远是不变的 (Rotation Invariant)。
        inv = torch.norm(x, dim=-1)  
        
        # 将求出的标量特征输入普通的 MLP 中学习更高阶的不变特征
        out: torch.Tensor = self.mlp(inv)
        return out

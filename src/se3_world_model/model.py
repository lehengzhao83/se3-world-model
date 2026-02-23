import torch
import torch.nn as nn
from se3_world_model.components import SE3Encoder
from se3_world_model.forces import ContextualForceGenerator, inject_global_vectors
from se3_world_model.layers import VNLinear, VNResBlock, VNInvariant

"""
================================================================================
模型接口与张量维度说明 (Input/Output & Dimensions)
================================================================================

输入 (Inputs):
1. x_history: 历史位置序列
2. v_history: 历史速度序列
3. explicit_vectors: 显式全局特征向量（如重力等，能够保持 SE(3) 等变性的矢量）
4. implicit_context: 隐式上下文特征（如破坏对称性的标量风力等）

输出 (Outputs):
1. pred_v: 预测的下一个时间步的每个点的速度
2. z_next: 融合了动力学和上下文信息的下一时间步隐变量

x, v, z 的维度 (Dimensions of x, v, z):
- x (位置/坐标):
  - 原始输入 x_history: [B, H, N, 3] 
    (B: Batch Size, H: History Length, N: Num Points, 3: 3D坐标 xyz)
  - 局部坐标 x_local (重排并去质心后): [B, N, H, 3]
- v (速度):
  - 原始输入 v_history: [B, H, N, 3]
    (B: Batch Size, H: History Length, N: Num Points, 3: 3D速度 xyz)
  - 重排后的速度 v_perm: [B, N, H, 3]
- z (隐变量):
  - 编码后的初始隐变量 z: [B, Latent_Dim, 3] 
    (Latent_Dim 是隐藏层通道数，每个通道都是一个3D向量，以此保持严格的SE(3)等变性)
  - 预测的下一时刻隐变量 z_next: [B, Latent_Dim, 3]

================================================================================
"""

class SE3RigidDecoder(nn.Module):
    """
    SE(3) 刚体解码器：将等变隐变量 z 解码为刚体运动的参数（质心速度变化量 和 旋转角轴）。
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        # 使用等变残差块和线性层，将高维隐特征降维到 2 个 3D 向量
        # 这 2 个向量分别代表：
        # 1. dv_cm: 质心平移速度的改变量 (Delta velocity of Center of Mass)
        # 2. theta: 旋转的角轴表示 (Angle-axis representation for rotation)
        self.net = nn.Sequential(
            VNResBlock(latent_dim),
            VNResBlock(latent_dim),
            VNLinear(latent_dim, 2)  # 将通道数从 latent_dim 映射到 2
        )

    def forward(self, z):
        # 输入 z 维度: [B, Latent_Dim, 3]
        # 输出维度: [B, 2, 3]
        return self.net(z) 

class SE3WorldModel(nn.Module):
    """
    SE(3) 等变世界模型主体。
    负责接收历史点云序列，结合物理上下文信息，预测系统未来的刚体运动状态。
    """
    def __init__(self, num_points=64, latent_dim=128, num_global_vectors=1, context_dim=3, history_len=1):
        super().__init__()
        self.history_len = history_len
        # 输入通道数为位置和速度的历史长度之和：每个历史步有 1 个位置向量和 1 个速度向量
        in_channels = 2 * history_len 
        
        # 1. 编码器：将点云及其历史速度编码为全局隐状态 z
        self.encoder = SE3Encoder(in_channels=in_channels, latent_dim=latent_dim)
        
        # 2. 动力学输入映射：将隐状态与显式全局向量（如重力）拼接后映射回 latent_dim 通道数
        self.dyn_input = VNLinear(latent_dim + num_global_vectors, latent_dim)
        
        # 3. 动力学主干网络：用于在隐空间中模拟系统演化的等变网络
        self.dyn_backbone = nn.Sequential(
            VNResBlock(latent_dim), 
            VNResBlock(latent_dim), 
            VNResBlock(latent_dim)
        )
        
        # 4. 上下文适配器：处理隐式环境信息（这类信息可能打破对称性，例如风力），输出非等变的修正向量
        self.context_adapter = ContextualForceGenerator(context_dim, hidden_dim=64)
        
        # 5. 动力学融合层：将等变动力学特征与非等变的上下文修正特征进行融合
        self.dyn_fusion = VNLinear(latent_dim + 1, latent_dim)
        
        # 6. 刚体解码器：从融合后的隐状态中预测下一步的刚体运动参数
        self.decoder = SE3RigidDecoder(latent_dim)

    def forward(self, x_history, v_history, explicit_vectors, implicit_context):
        # 获取输入形状：B=Batch, H=History, N=Num Points, _=3 (3D坐标维度)
        B, H, N, _ = x_history.shape
        
        # 校验传入数据的历史长度是否与模型初始化时一致
        if H != self.history_len:
            raise ValueError("History length mismatch")

        # 将 H (历史) 和 N (点数) 维度交换，方便后续按点聚合处理特征
        # 变换前: [B, H, N, 3] -> 变换后 (x_perm, v_perm): [B, N, H, 3]
        x_perm = x_history.permute(0, 2, 1, 3)
        v_perm = v_history.permute(0, 2, 1, 3)
        
        # 恢复平移不变性：将绝对坐标转换为相对质心的局部坐标
        # 计算所有点在所有历史时刻的中心位置 (x_center 维度: [B, 1, H, 3])
        x_center = x_perm.mean(dim=1, keepdim=True)
        # 减去质心，得到局部坐标系下的位置特征
        x_local = x_perm - x_center
        
        # 将局部位置和速度在特征通道维度 (dim=2) 拼接，作为编码器输入
        # x_in 维度: [B, N, 2*H, 3] (2*H 等于 in_channels)
        x_in = torch.cat([x_local, v_perm], dim=2)
        
        # 通过编码器提取全局隐变量 z
        # z 维度: [B, latent_dim, 3]
        z = self.encoder(x_in)

        # 注入显式全局向量（如重力等，此类向量需要参与等变运算）
        # z_aug 维度: [B, latent_dim + num_global_vectors, 3]
        z_aug = inject_global_vectors(z, explicit_vectors)
        
        # 隐空间动力学演化，提取物理动力学特征
        # z_pred_raw 维度: [B, latent_dim, 3]
        z_pred_raw = self.dyn_backbone(self.dyn_input(z_aug))
        
        # 生成基于非等变上下文的修正向量（如受风力影响，破坏了旋转对称性）
        # correction 维度: [B, 1, 3]
        correction = self.context_adapter(implicit_context)
        
        # 将动力学特征和上下文修正拼接后进行特征融合，得到下一时刻的隐变量预测
        # z_next 维度: [B, latent_dim, 3]
        z_next = self.dyn_fusion(torch.cat([z_pred_raw, correction], dim=1))

        # 解码下一时刻的刚体变换参数
        # rigid_params 维度: [B, 2, 3]
        rigid_params = self.decoder(z_next)
        
        # 分离出 质心速度改变量(dv_cm) 和 旋转角轴(theta)
        dv_cm = rigid_params[:, 0:1, :] # 维度: [B, 1, 3]
        theta = rigid_params[:, 1:2, :] # 维度: [B, 1, 3]

        # 获取当前（也就是最后一步历史）的绝对位置和速度
        x_curr = x_history[:, -1] # 维度: [B, N, 3]
        v_curr = v_history[:, -1] # 维度: [B, N, 3]
        
        # 计算当前时刻的质心
        center = x_curr.mean(dim=1, keepdim=True) # 维度: [B, 1, 3]
        # 获取当前时刻每个点相对于质心的局部向量 r
        r = x_curr - center # 维度: [B, N, 3]
        
        # 计算当前时刻的整体质心速度
        v_cm = v_curr.mean(dim=1, keepdim=True) # 维度: [B, 1, 3]
        # 根据模型预测的改变量，计算下一时刻的质心速度
        next_v_cm = v_cm + dv_cm # 维度: [B, 1, 3]
        
        # ==================== 旋转矩阵计算 (Rodrigues' rotation formula) ====================
        # 计算旋转角 (即 theta 向量的模长)
        angle = torch.norm(theta, dim=-1, keepdim=True) # 维度: [B, 1, 1]
        # 计算旋转轴 (即归一化的 theta 向量)
        axis = theta / (angle + 1e-6) # 维度: [B, 1, 3]，加 1e-6 防止除零
        
        # 构造叉乘矩阵 K (用于罗德里格斯旋转公式)
        K = torch.zeros(B, 3, 3, device=x_curr.device)
        K[:, 0, 1] = -axis[:, 0, 2]
        K[:, 0, 2] = axis[:, 0, 1]
        K[:, 1, 0] = axis[:, 0, 2]
        K[:, 1, 2] = -axis[:, 0, 0]
        K[:, 2, 0] = -axis[:, 0, 1]
        K[:, 2, 1] = axis[:, 0, 0]
        
        # 构造单位矩阵 I
        I = torch.eye(3, device=x_curr.device).unsqueeze(0).expand(B, 3, 3)
        sin_a = torch.sin(angle) # 维度: [B, 1, 1]
        cos_a = torch.cos(angle) # 维度: [B, 1, 1]
        
        # 根据罗德里格斯公式计算最终的 3x3 旋转矩阵 R
        # 公式: R = I + sin(a)K + (1 - cos(a))K^2
        R = I + sin_a * K + (1 - cos_a) * torch.bmm(K, K) # 维度: [B, 3, 3]
        # ==================================================================================
        
        # 将当前的局部向量 r 进行旋转，得到下一时刻的相对位置 r_rotated
        # r 维度: [B, N, 3], R^T 维度: [B, 3, 3], bmm 结果: [B, N, 3]
        r_rotated = torch.bmm(r, R.transpose(1, 2))
        
        # 计算每个点下一时刻的预测速度
        # 预测速度 = 预测的质心速度 + 由于刚体旋转产生的位置变化差分 (旋转后的相对位置 - 旋转前的相对位置)
        pred_v = next_v_cm + (r_rotated - r)
        
        # 返回最终预测的各个点速度和下一步的隐变量特征
        return pred_v, z_next

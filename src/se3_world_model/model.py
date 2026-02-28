import torch
import torch.nn as nn

# 引入重构后的各类算子
from se3_world_model.components import SE3Encoder
from se3_world_model.forces import EquivariantContextModulator, inject_global_vectors
from se3_world_model.layers import VNLinear, VNGatedBlock, safe_norm

class SE3RigidDecoder(nn.Module):
    # 此处保持原有代码不变
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            VNGatedBlock(latent_dim), # 使用新的 GatedBlock 替换 ResBlock
            VNGatedBlock(latent_dim),
            VNLinear(latent_dim, 2)
        )
    def forward(self, z):
        return self.net(z)

class SE3WorldModel(nn.Module):
    def __init__(self, num_points=64, latent_dim=128, num_global_vectors=1, context_dim=3, history_len=1):
        super().__init__()
        self.history_len = history_len
        
        # 输入通道变更为 3： 局部坐标(1) + 速度(1) + 接触力(1)
        in_channels = 3 
        
        # 1. 带有 Temporal Attention 的等变编码器
        self.encoder = SE3Encoder(in_channels=in_channels, latent_dim=latent_dim, num_heads=4)
        
        # 2. 动力学输入映射：拼接归一化后的全局方向向量
        self.dyn_input = VNLinear(latent_dim + num_global_vectors, latent_dim)
        
        # 3. 动力学主干：使用 Gated Block 允许通道间基于不变量产生复杂交互
        self.dyn_backbone = nn.Sequential(
            VNGatedBlock(latent_dim), 
            VNGatedBlock(latent_dim), 
            VNGatedBlock(latent_dim)
        )
        
        # 4. 上下文调制器：将原本的 context_dim 加上分解出的全局向量模长(标量)
        total_scalar_dim = context_dim + num_global_vectors
        self.context_modulator = EquivariantContextModulator(total_scalar_dim, latent_dim)
        
        # 5. 解码器
        self.decoder = SE3RigidDecoder(latent_dim)

    # 注意前向传播增加了 f_history 参数 (来自 dataset)
    def forward(self, x_history, v_history, f_history, explicit_vectors, implicit_context):
        B, H, N, _ = x_history.shape
        
        # [B, N, H, 3]
        x_perm = x_history.permute(0, 2, 1, 3)
        v_perm = v_history.permute(0, 2, 1, 3)
        f_perm = f_history.permute(0, 2, 1, 3)
        
        x_center = x_perm.mean(dim=1, keepdim=True)
        x_local = x_perm - x_center
        
        # 沿着新增加的通道维度拼接 -> x_in 维度: [B, N, H, 3, 3]
        x_in = torch.stack([x_local, v_perm, f_perm], dim=3)
        
        # 编码器在时间轴上做 Attention 聚合
        z = self.encoder(x_in) # [B, latent_dim, 3]

        # === 核心重构：宏观风力向量的标量分解 ===
        # 计算全局向量（如风力）的模长（标量）和方向（单位向量）
        vec_mags = safe_norm(explicit_vectors, dim=-1) # [B, N_vecs, 1]
        vec_dirs = explicit_vectors / vec_mags         # [B, N_vecs, 3]
        
        # 将方向向量（等变）拼接到隐变量中
        z_aug = inject_global_vectors(z, vec_dirs)
        
        # 提取等变动力学特征
        z_pred_raw = self.dyn_backbone(self.dyn_input(z_aug))
        
        # === 核心重构：等变调制替换暴力拼接 ===
        # 将环境上下文与风力大小组合成一个纯标量特征向量
        combined_context = torch.cat([implicit_context, vec_mags.squeeze(-1)], dim=-1) # [B, context_dim + N_vecs]
        
        # 使用纯标量权重对等变动力学特征进行调制，严格维持等变性
        z_next = self.context_modulator(combined_context, z_pred_raw)

        # 解码刚体变换参数
        rigid_params = self.decoder(z_next)
        
        dv_cm = rigid_params[:, 0:1, :]
        # 放大旋转轴角响应，跳出“只平移”的局部最优
        theta = rigid_params[:, 1:2, :] * 10.0

        # ---------- 以下罗德里格斯旋转公式部分保持你的原代码不变 ----------
        x_curr = x_history[:, -1]
        v_curr = v_history[:, -1]
        
        center = x_curr.mean(dim=1, keepdim=True)
        r = x_curr - center
        v_cm = v_curr.mean(dim=1, keepdim=True)
        next_v_cm = v_cm + dv_cm
        
        angle = torch.norm(theta, dim=-1, keepdim=True)
        axis = theta / (angle + 1e-6)
        
        K = torch.zeros(B, 3, 3, device=x_curr.device)
        K[:, 0, 1] = -axis[:, 0, 2]
        K[:, 0, 2] = axis[:, 0, 1]
        K[:, 1, 0] = axis[:, 0, 2]
        K[:, 1, 2] = -axis[:, 0, 0]
        K[:, 2, 0] = -axis[:, 0, 1]
        K[:, 2, 1] = axis[:, 0, 0]
        
        I = torch.eye(3, device=x_curr.device).unsqueeze(0).expand(B, 3, 3)
        sin_a = torch.sin(angle)
        cos_a = torch.cos(angle)
        
        R = I + sin_a * K + (1 - cos_a) * torch.bmm(K, K)
        r_rotated = torch.bmm(r, R.transpose(1, 2))
        
        pred_v = next_v_cm + (r_rotated - r)
        # -------------------------------------------------------------
        
        return pred_v, z_next

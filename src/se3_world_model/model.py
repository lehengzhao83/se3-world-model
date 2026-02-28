import torch
import torch.nn as nn

# 引入重构后的各类算子
from se3_world_model.components import SE3Encoder
from se3_world_model.forces import EquivariantContextModulator, inject_global_vectors
from se3_world_model.layers import VNLinear, VNGatedBlock, safe_norm

class SE3RigidDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            VNGatedBlock(latent_dim), 
            VNGatedBlock(latent_dim),
            VNLinear(latent_dim, 2)
        )
    def forward(self, z):
        return self.net(z)

class SE3WorldModel(nn.Module):
    def __init__(self, num_points=64, latent_dim=128, num_global_vectors=1, context_dim=3, history_len=1):
        super().__init__()
        self.history_len = history_len
        
        # 输入通道变更为 2： 局部坐标(1) + 速度(1) （去掉了接触力）
        in_channels = 2 
        
        self.encoder = SE3Encoder(in_channels=in_channels, latent_dim=latent_dim, num_heads=4)
        
        self.dyn_input = VNLinear(latent_dim + num_global_vectors, latent_dim)
        
        self.dyn_backbone = nn.Sequential(
            VNGatedBlock(latent_dim), 
            VNGatedBlock(latent_dim), 
            VNGatedBlock(latent_dim)
        )
        
        # 上下文维度增加 1，用于显式传入绝对高度 (Z坐标)
        total_scalar_dim = context_dim + num_global_vectors + 1
        self.context_modulator = EquivariantContextModulator(total_scalar_dim, latent_dim)
        
        self.decoder = SE3RigidDecoder(latent_dim)

    def forward(self, x_history, v_history, explicit_vectors, implicit_context):
        B, H, N, _ = x_history.shape
        
        x_perm = x_history.permute(0, 2, 1, 3)
        v_perm = v_history.permute(0, 2, 1, 3)
        
        # 提取最后一步质心的 Z 坐标作为距地面的高度
        z_height = x_perm[:, :, -1, 2].mean(dim=1, keepdim=True) # [B, 1]
        
        x_center = x_perm.mean(dim=1, keepdim=True)
        x_local = x_perm - x_center
        
        # 只拼接局部坐标和速度 (x_in 维度: [B, N, H, 2, 3])
        x_in = torch.stack([x_local, v_perm], dim=3)
        
        z = self.encoder(x_in) # [B, latent_dim, 3]

        # 分解显式向量（如重力）
        vec_mags = safe_norm(explicit_vectors, dim=-1) # [B, N_vecs, 1]
        vec_dirs = explicit_vectors / vec_mags         # [B, N_vecs, 3]
        
        z_aug = inject_global_vectors(z, vec_dirs)
        
        z_pred_raw = self.dyn_backbone(self.dyn_input(z_aug))
        
        # 将环境上下文、向量模长、和绝对高度合并
        combined_context = torch.cat([implicit_context, vec_mags.squeeze(-1), z_height], dim=-1) # [B, context_dim + N_vecs + 1]
        
        z_next = self.context_modulator(combined_context, z_pred_raw)

        # 解码刚体变换参数
        rigid_params = self.decoder(z_next)
        
        # ====== 就是这里！！！ ======
        dv_cm = rigid_params[:, 0:1, :]
        theta = rigid_params[:, 1:2, :] * 10.0
        # ==========================

        # ---------- 罗德里格斯旋转公式 ----------
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
        
        return pred_v, z_next

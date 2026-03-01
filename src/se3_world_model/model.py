import torch
import torch.nn as nn
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
        in_channels = 2 
        
        self.encoder = SE3Encoder(in_channels=in_channels, latent_dim=latent_dim, num_heads=4)
        self.dyn_input = VNLinear(latent_dim + num_global_vectors, latent_dim)
        self.dyn_backbone = nn.Sequential(
            VNGatedBlock(latent_dim), 
            VNGatedBlock(latent_dim), 
            VNGatedBlock(latent_dim)
        )
        
        total_scalar_dim = context_dim + num_global_vectors + 1
        self.context_modulator = EquivariantContextModulator(total_scalar_dim, latent_dim)
        self.decoder = SE3RigidDecoder(latent_dim)

    def forward(self, x_history, v_history, explicit_vectors, implicit_context, vel_std=None, pos_mean=None, pos_std=None):
        B, H, N, _ = x_history.shape
        x_perm = x_history.permute(0, 2, 1, 3)
        v_perm = v_history.permute(0, 2, 1, 3)
        
        if pos_mean is not None and pos_std is not None:
            x_real = x_perm * pos_std.to(x_perm.device) + pos_mean.to(x_perm.device)
            z_height = x_real[:, :, -1, 2].mean(dim=1, keepdim=True)
        else:
            z_height = x_perm[:, :, -1, 2].mean(dim=1, keepdim=True)
        
        x_center = x_perm.mean(dim=1, keepdim=True)
        x_local = x_perm - x_center
        x_in = torch.stack([x_local, v_perm], dim=3)
        z = self.encoder(x_in)

        vec_mags = safe_norm(explicit_vectors, dim=-1)
        vec_dirs = explicit_vectors / vec_mags         
        
        z_aug = inject_global_vectors(z, vec_dirs)
        z_pred_raw = self.dyn_backbone(self.dyn_input(z_aug))
        
        combined_context = torch.cat([implicit_context, vec_mags.squeeze(-1), z_height], dim=-1)
        z_next = self.context_modulator(combined_context, z_pred_raw)

        rigid_params = self.decoder(z_next)
        
        dv_cm = rigid_params[:, 0:1, :]
        theta = rigid_params[:, 1:2, :] * 10.0

        x_curr = x_history[:, -1]
        v_curr = v_history[:, -1]
        
        center = x_curr.mean(dim=1, keepdim=True)
        r = x_curr - center
        v_cm = v_curr.mean(dim=1, keepdim=True)
        next_v_cm = v_cm + dv_cm
        
        angle = torch.norm(theta, dim=-1, keepdim=True).clamp(min=1e-6)
        axis = theta / angle
        
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
        rot_displacement = r_rotated - r
        
        if vel_std is not None:
            rot_displacement = rot_displacement / vel_std.to(rot_displacement.device)
            
        # 1. 预测无约束状态下的速度
        pred_v_unconstrained = next_v_cm + rot_displacement
        
        # 2. 【核心重构：移除强硬的 SI 求解器，打通碰撞学习的梯度网络】
        pred_v = pred_v_unconstrained
        
        # 返回时多暴露 next_v_cm 和 R 给外部用于绝对刚性 Rollout
        return pred_v, z_next, next_v_cm, R

import torch
import torch.nn as nn
from se3_world_model.components import SE3Encoder
from se3_world_model.forces import ContextualForceGenerator, inject_global_vectors
from se3_world_model.layers import VNLinear, VNResBlock, VNInvariant

class SE3RigidDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            VNResBlock(latent_dim),
            VNResBlock(latent_dim),
            VNLinear(latent_dim, 2) 
        )

    def forward(self, z):
        return self.net(z) 

class SE3WorldModel(nn.Module):
    def __init__(self, num_points=64, latent_dim=128, num_global_vectors=1, context_dim=3, history_len=1):
        super().__init__()
        self.history_len = history_len
        
        # === 核心逻辑：通道堆叠 ===
        # 输入维度: [B, H, N, 3] -> 堆叠后 [B, N, 2*H, 3]
        in_channels = 2 * history_len 
        
        self.encoder = SE3Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.dyn_input = VNLinear(latent_dim + num_global_vectors, latent_dim)
        self.dyn_backbone = nn.Sequential(VNResBlock(latent_dim), VNResBlock(latent_dim), VNResBlock(latent_dim))
        self.context_adapter = ContextualForceGenerator(context_dim, hidden_dim=64)
        self.dyn_fusion = VNLinear(latent_dim + 1, latent_dim)
        self.decoder = SE3RigidDecoder(latent_dim)

    def forward(self, x_history, v_history, explicit_vectors, implicit_context):
        # x_history: [B, H, N, 3]
        B, H, N, _ = x_history.shape
        
        # 维度检查 (非常重要，防止配置错误)
        if H != self.history_len:
            raise ValueError(f"Input history length {H} does not match model config {self.history_len}")

        # 1. 维度变换: [B, H, N, 3] -> [B, N, H, 3]
        x_perm = x_history.permute(0, 2, 1, 3)
        v_perm = v_history.permute(0, 2, 1, 3)
        
        # 2. 通道拼接: [B, N, 2H, 3]
        x_in = torch.cat([x_perm, v_perm], dim=2)
        
        # Encoder
        z = self.encoder(x_in)

        # Dynamics
        z_aug = inject_global_vectors(z, explicit_vectors)
        z_pred_raw = self.dyn_backbone(self.dyn_input(z_aug))
        correction = self.context_adapter(implicit_context)
        z_next = self.dyn_fusion(torch.cat([z_pred_raw, correction], dim=1))

        # Decode
        rigid_params = self.decoder(z_next)
        acc_lin = rigid_params[:, 0:1, :] 
        acc_ang = rigid_params[:, 1:2, :] 

        # 物理积分：总是基于当前时刻 (History 的最后一帧)
        x_curr = x_history[:, -1] # [B, N, 3]
        v_curr = v_history[:, -1] # [B, N, 3]
        
        center = x_curr.mean(dim=1, keepdim=True)
        r = x_curr - center
        acc_ang_expanded = acc_ang.expand(-1, N, -1)
        delta_v_rot = torch.cross(acc_ang_expanded, r, dim=-1)
        
        pred_v = v_curr + acc_lin + delta_v_rot
        
        return pred_v, z_next

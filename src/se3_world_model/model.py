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
        in_channels = 2 * history_len 
        
        self.encoder = SE3Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.dyn_input = VNLinear(latent_dim + num_global_vectors, latent_dim)
        self.dyn_backbone = nn.Sequential(VNResBlock(latent_dim), VNResBlock(latent_dim), VNResBlock(latent_dim))
        self.context_adapter = ContextualForceGenerator(context_dim, hidden_dim=64)
        self.dyn_fusion = VNLinear(latent_dim + 1, latent_dim)
        self.decoder = SE3RigidDecoder(latent_dim)

    def forward(self, x_history, v_history, explicit_vectors, implicit_context):
        B, H, N, _ = x_history.shape
        if H != self.history_len:
            raise ValueError("History length mismatch")

        x_perm = x_history.permute(0, 2, 1, 3)
        v_perm = v_history.permute(0, 2, 1, 3)
        
        # 恢复平移不变性：将绝对坐标转为相对质心的局部坐标
        x_center = x_perm.mean(dim=1, keepdim=True)
        x_local = x_perm - x_center
        
        x_in = torch.cat([x_local, v_perm], dim=2)
        z = self.encoder(x_in)

        z_aug = inject_global_vectors(z, explicit_vectors)
        z_pred_raw = self.dyn_backbone(self.dyn_input(z_aug))
        correction = self.context_adapter(implicit_context)
        z_next = self.dyn_fusion(torch.cat([z_pred_raw, correction], dim=1))

        rigid_params = self.decoder(z_next)
        dv_cm = rigid_params[:, 0:1, :] 
        theta = rigid_params[:, 1:2, :] 

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

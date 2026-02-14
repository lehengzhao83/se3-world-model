import torch
import torch.nn as nn
from se3_world_model.components import SE3Encoder
from se3_world_model.forces import ContextualForceGenerator, inject_global_vectors
from se3_world_model.layers import VNLinear, VNResBlock, VNInvariant

# 定义一个新的 RigidDecoder，只预测全局刚体参数
class SE3RigidDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        # 只需要预测两个全局向量：线加速度(a) 和 角加速度(alpha)
        # 输入 Latent: [B, C, 3] -> 输出: [B, 2, 3]
        self.net = nn.Sequential(
            VNResBlock(latent_dim),
            VNResBlock(latent_dim),
            VNLinear(latent_dim, 2) # Channel=2: Channel 0 -> acc, Channel 1 -> angular_acc
        )

    def forward(self, z):
        # z: [B, C, 3]
        out = self.net(z) # [B, 2, 3]
        return out

class SE3WorldModel(nn.Module):
    def __init__(self, num_points=1024, latent_dim=128, num_global_vectors=1, context_dim=3):
        super().__init__()
        self.encoder = SE3Encoder(in_channels=2, latent_dim=latent_dim)
        self.dyn_input = VNLinear(latent_dim + num_global_vectors, latent_dim)
        self.dyn_backbone = nn.Sequential(VNResBlock(latent_dim), VNResBlock(latent_dim), VNResBlock(latent_dim))
        self.context_adapter = ContextualForceGenerator(context_dim, hidden_dim=64)
        self.dyn_fusion = VNLinear(latent_dim + 1, latent_dim)
        self.decoder = SE3RigidDecoder(latent_dim)

    def forward(self, x, v, explicit_vectors, implicit_context):
        # 1. Encode
        x_in = torch.stack([x, v], dim=2)
        z = self.encoder(x_in)

        # 2. Dynamics
        z_aug = inject_global_vectors(z, explicit_vectors)
        z_pred_raw = self.dyn_backbone(self.dyn_input(z_aug))
        correction = self.context_adapter(implicit_context)
        z_next = self.dyn_fusion(torch.cat([z_pred_raw, correction], dim=1))

        # 3. Decode Rigid Parameters
        # rigid_params: [B, 2, 3]
        # acc_lin = rigid_params[:, 0, :] (线加速度)
        # acc_ang = rigid_params[:, 1, :] (角加速度)
        rigid_params = self.decoder(z_next)
        acc_lin = rigid_params[:, 0:1, :] # [B, 1, 3]
        acc_ang = rigid_params[:, 1:2, :] # [B, 1, 3]

        # 4. Apply Rigid Body Kinematics to recover per-point delta v
        # v_i_new = v_i_old + acc_lin + acc_ang x r_i
        # r_i = x_i - center_of_mass
        
        center = x.mean(dim=1, keepdim=True) # [B, 1, 3]
        r = x - center # [B, N, 3]
        
        # 叉乘: acc_ang x r
        # 此时 acc_ang 需要广播到 [B, N, 3]
        acc_ang_expanded = acc_ang.expand(-1, x.shape[1], -1)
        delta_v_rot = torch.cross(acc_ang_expanded, r, dim=-1)
        
        # 总速度增量
        delta_v = acc_lin + delta_v_rot
        
        # 残差连接
        pred_v = v + delta_v
        
        return pred_v, z_next

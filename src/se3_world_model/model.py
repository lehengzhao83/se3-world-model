import torch
import torch.nn as nn

from se3_world_model.components import SE3Decoder, SE3Encoder
from se3_world_model.forces import ContextualForceGenerator, inject_global_vectors
from se3_world_model.layers import VNLinear, VNResBlock


class SE3WorldModel(nn.Module):
    def __init__(
        self,
        num_points: int = 1024,
        latent_dim: int = 128, # 保持与 train.py 一致 (如果是128请改这里)
        num_global_vectors: int = 1,
        context_dim: int = 3
    ) -> None:
        super().__init__()

        # Encoder: Takes [Pos, Vel] -> 2 channels
        self.encoder = SE3Encoder(in_channels=2, latent_dim=latent_dim)

        # Dynamics Backbone
        self.dyn_input = VNLinear(latent_dim + num_global_vectors, latent_dim)
        self.dyn_backbone = nn.Sequential(
            VNResBlock(latent_dim),
            VNResBlock(latent_dim),
            VNResBlock(latent_dim)
        )
        
        # Context Adapter
        self.context_adapter = ContextualForceGenerator(context_dim, hidden_dim=64)
        self.dyn_fusion = VNLinear(latent_dim + 1, latent_dim)

        # Decoder: Latent -> Velocity Delta (Acceleration)
        self.decoder = SE3Decoder(latent_dim, num_points)

    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        explicit_vectors: torch.Tensor,
        implicit_context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input: Normalized Pos, Normalized Vel
        Output: Normalized Predicted Vel
        """
        
        # 1. Encode [B, N, 2, 3]
        x_in = torch.stack([x, v], dim=2)
        z = self.encoder(x_in)

        # 2. Dynamics
        z_aug = inject_global_vectors(z, explicit_vectors)
        z_curr = self.dyn_input(z_aug)
        z_pred_raw = self.dyn_backbone(z_curr)

        correction = self.context_adapter(implicit_context)
        z_combined = torch.cat([z_pred_raw, correction], dim=1)
        z_next = self.dyn_fusion(z_combined)

        # 3. Decode -> Predicted Delta (Acceleration)
        # 模型现在只需要预测“变化量”，这比预测全量容易得多
        pred_delta_v = self.decoder(z_next)

        # === 核心修改：残差连接 (Residual Connection) ===
        # v_{t+1} = v_t + pred_acceleration
        # 这样模型初始状态就是“惯性保持”(Identity)，Loss 会瞬间降到 0.15 以下
        pred_v = v + pred_delta_v 

        return pred_v, z_next

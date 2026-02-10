import torch
import torch.nn as nn

from se3_world_model.components import SE3Decoder, SE3Encoder
from se3_world_model.forces import ContextualForceGenerator, inject_global_vectors
from se3_world_model.layers import VNLinear, VNResBlock


class SE3WorldModel(nn.Module):
    def __init__(
        self,
        num_points: int = 1024,
        latent_dim: int = 64,
        num_global_vectors: int = 1,  # e.g., Gravity
        context_dim: int = 3         # e.g., World Position
    ) -> None:
        super().__init__()

        # 1. Encoder (Raw points in, Latent state out)
        # in_channels=1 because input is raw xyz
        self.encoder = SE3Encoder(in_channels=1, latent_dim=latent_dim)

        # 2. Dynamics (Hybrid)

        # Step A: Explicit Global Vectors (Gravity)
        # Input projection accommodates latent state + global vectors
        self.dyn_input = VNLinear(latent_dim + num_global_vectors, latent_dim)

        # Step B: Main Backbone (Pure SE(3) Physics)
        self.dyn_backbone = nn.Sequential(
            VNResBlock(latent_dim),
            VNResBlock(latent_dim),
            VNResBlock(latent_dim)
        )

        # Step C: Implicit Data-Driven Fields
        # Adapter to learn non-equivariant forces from context
        self.context_adapter = ContextualForceGenerator(context_dim, hidden_dim=64)

        # Step D: Fusion (Backbone + Learned Force)
        # latent_dim (state) + 1 (force vector) -> latent_dim
        self.dyn_fusion = VNLinear(latent_dim + 1, latent_dim)

        # 3. Decoder (Latent state out, Point Cloud out)
        self.decoder = SE3Decoder(latent_dim, num_points)

    def forward(
        self,
        x: torch.Tensor,
        explicit_vectors: torch.Tensor,
        implicit_context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, 3] Input Point Cloud.
            explicit_vectors: [B, M, 3] Known vectors (Gravity/Wind) in Body Frame.
            implicit_context: [B, K] Unknown params (Position/Time) for data-driven adaptation.

        Returns:
            pred_x: [B, N, 3] Predicted Next Point Cloud.
            z_next: [B, Latent, 3] Predicted Latent State.
        """

        # 1. Encode
        z = self.encoder(x)  # [B, Latent, 3]

        # 2. Dynamics

        # A. Inject Explicit Forces
        z_aug = inject_global_vectors(z, explicit_vectors)
        z_curr: torch.Tensor = self.dyn_input(z_aug)

        # B. Evolve State (Equivariant)
        z_pred_raw: torch.Tensor = self.dyn_backbone(z_curr)

        # C. Inject Implicit Forces (Data-driven Residual)
        correction_force = self.context_adapter(implicit_context)  # [B, 1, 3]

        # D. Fuse Correction
        z_combined = torch.cat([z_pred_raw, correction_force], dim=1)
        z_next: torch.Tensor = self.dyn_fusion(z_combined)

        # 3. Decode
        pred_x = self.decoder(z_next)  # [B, N, 3]

        return pred_x, z_next

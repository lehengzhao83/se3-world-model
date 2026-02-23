import unittest
import torch
from se3_world_model.model import SE3WorldModel

class TestSE3WorldModel(unittest.TestCase):
    def test_forward_pass_and_shapes(self) -> None:
        # Config
        batch_size = 2
        history_len = 3
        num_points = 64
        latent_dim = 16
        num_global_vectors = 2  
        context_dim = 4        

        model = SE3WorldModel(
            num_points=num_points,
            latent_dim=latent_dim,
            num_global_vectors=num_global_vectors,
            context_dim=context_dim,
            history_len=history_len
        )

        # Mock Inputs: [B, H, N, 3]
        x_hist = torch.randn(batch_size, history_len, num_points, 3)
        v_hist = torch.randn(batch_size, history_len, num_points, 3)
        explicit_vecs = torch.randn(batch_size, num_global_vectors, 3)
        implicit_ctx = torch.randn(batch_size, context_dim)

        # Forward
        pred_v, z_next = model(x_hist, v_hist, explicit_vecs, implicit_ctx)

        # Assertions
        self.assertEqual(pred_v.shape, (batch_size, num_points, 3))
        self.assertEqual(z_next.shape, (batch_size, latent_dim, 3))

        print("Test passed: Forward pass shapes are correct.")

if __name__ == '__main__':
    unittest.main()

import unittest
import torch
from se3_world_model.model import SE3WorldModel
from se3_world_model.loss import GeometricConsistencyLoss

class TestSE3WorldModel(unittest.TestCase):
    def setUp(self) -> None:
        # 将测试参数提取为类的公共属性，方便多个测试用例复用
        self.batch_size = 2
        self.history_len = 3
        self.num_points = 64
        self.latent_dim = 16
        self.num_global_vectors = 2  
        self.context_dim = 4        

        self.model = SE3WorldModel(
            num_points=self.num_points,
            latent_dim=self.latent_dim,
            num_global_vectors=self.num_global_vectors,
            context_dim=self.context_dim,
            history_len=self.history_len
        )

    def test_forward_pass_and_shapes(self) -> None:
        """测试前向传播的维度是否符合预期"""
        x_hist = torch.randn(self.batch_size, self.history_len, self.num_points, 3)
        v_hist = torch.randn(self.batch_size, self.history_len, self.num_points, 3)
        explicit_vecs = torch.randn(self.batch_size, self.num_global_vectors, 3)
        implicit_ctx = torch.randn(self.batch_size, self.context_dim)

        pred_v, z_next = self.model(x_hist, v_hist, explicit_vecs, implicit_ctx)

        self.assertEqual(pred_v.shape, (self.batch_size, self.num_points, 3))
        self.assertEqual(z_next.shape, (self.batch_size, self.latent_dim, 3))
        print("Test passed: Forward pass shapes are correct.")

    def test_backward_pass_no_nan_with_zero_inputs(self) -> None:
        """
        核心测试：验证反向传播在极端零输入下的稳定性。
        确保 safe_norm 机制有效，杜绝求导产生 NaN（梯度爆炸）。
        """
        # 故意构造全 0 的物理输入（物体完全静止且不受力）
        x_hist = torch.zeros(self.batch_size, self.history_len, self.num_points, 3)
        v_hist = torch.zeros(self.batch_size, self.history_len, self.num_points, 3)
        explicit_vecs = torch.zeros(self.batch_size, self.num_global_vectors, 3)
        implicit_ctx = torch.zeros(self.batch_size, self.context_dim)
        
        # 目标值也设为全 0
        target_v = torch.zeros(self.batch_size, self.num_points, 3)
        target_x = torch.zeros(self.batch_size, self.num_points, 3)

        self.model.train()
        pred_v, _ = self.model(x_hist, v_hist, explicit_vecs, implicit_ctx)
        
        # 模拟物理积分计算 next_x
        last_x = x_hist[:, -1]
        next_x = last_x + pred_v
        
        # 实例化我们在 loss.py 中修复过的损失函数
        criterion = GeometricConsistencyLoss()
        loss, _, _, _ = criterion(next_x, target_x, pred_v, target_v)
        
        # 触发反向传播计算梯度
        loss.backward()

        # 遍历模型的所有参数，断言没有任何一个参数的梯度包含 NaN
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_nan = torch.isnan(param.grad).any().item()
                self.assertFalse(has_nan, f"Gradient explosion (NaN) detected in parameter: {name}")
                
        print("Test passed: Backward pass handles zero-vectors perfectly without NaNs.")

if __name__ == '__main__':
    unittest.main()

import torch
import torch.nn as nn

class GeometricConsistencyLoss(nn.Module):
    """
    混合物理损失函数：轨迹 + 刚性 + 能量约束
    """
    def __init__(self, lambda_rigid: float = 2.0, lambda_energy: float = 0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_rigid = lambda_rigid
        self.lambda_energy = lambda_energy

    def forward(self, pred_x: torch.Tensor, target_x: torch.Tensor, 
                pred_v: torch.Tensor, target_v: torch.Tensor):
        """
        Args:
            pred_x: [B, N, 3] 预测位置
            target_x: [B, N, 3] 真实位置
            pred_v: [B, N, 3] 预测速度
            target_v: [B, N, 3] 真实速度
        """
        # 1. 轨迹误差 (Trajectory Loss)
        traj_loss = self.mse(pred_x, target_x)
        
        # 2. 刚性误差 (Rigidity Loss)
        pred_dist = torch.cdist(pred_x, pred_x, p=2)
        target_dist = torch.cdist(target_x, target_x, p=2)
        rigid_loss = self.mse(pred_dist, target_dist)
        
        # 3. 能量/动能一致性 (Energy Consistency Loss)
        # 动能 K = 0.5 * m * v^2。假设质量均匀，只需约束 v^2 的一致性
        # 这能防止物体凭空加速或减速
        pred_k = torch.sum(pred_v**2, dim=-1)   # [B, N]
        target_k = torch.sum(target_v**2, dim=-1) # [B, N]
        energy_loss = self.mse(pred_k, target_k)
        
        # 总 Loss
        total_loss = traj_loss + \
                     self.lambda_rigid * rigid_loss + \
                     self.lambda_energy * energy_loss
        
        return total_loss, traj_loss, rigid_loss, energy_loss

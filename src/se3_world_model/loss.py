import torch
import torch.nn as nn

class GeometricConsistencyLoss(nn.Module):
    """
    混合物理损失函数：
    1. MSE Loss: 保证轨迹位置准确。
    2. Rigidity Loss (刚性约束): 保证物体在运动过程中不散架、不变形。
    """
    def __init__(self, lambda_rigid: float = 0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_rigid = lambda_rigid

    def forward(self, pred_x: torch.Tensor, target_x: torch.Tensor):
        """
        Args:
            pred_x: [B, N, 3] 预测点云
            target_x: [B, N, 3] 真实点云
        """
        # 1. 轨迹误差 (MSE)
        traj_loss = self.mse(pred_x, target_x)
        
        # 2. 刚性误差 (Rigidity / Isometric Loss)
        # 计算预测点云内部两两点之间的距离矩阵
        # P: [B, N, 3] -> Dist: [B, N, N]
        pred_dist = torch.cdist(pred_x, pred_x, p=2)
        target_dist = torch.cdist(target_x, target_x, p=2)
        
        # 强制要求：预测后的形状内部距离 = 真实形状内部距离
        # 这能有效防止点云“散开”或“缩成一团”
        rigid_loss = self.mse(pred_dist, target_dist)
        
        # 总 Loss
        total_loss = traj_loss + self.lambda_rigid * rigid_loss
        
        return total_loss, traj_loss, rigid_loss

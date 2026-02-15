import torch
import torch.nn as nn

class GeometricConsistencyLoss(nn.Module):
    def __init__(self, lambda_rigid: float = 2.0, lambda_energy: float = 0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.lambda_rigid = lambda_rigid
        self.lambda_energy = lambda_energy

    def forward(self, pred_x: torch.Tensor, target_x: torch.Tensor, 
                pred_v: torch.Tensor, target_v: torch.Tensor):
        
        # 1. 轨迹误差 (MSE)
        traj_loss = self.mse(pred_x, target_x)
        
        # 2. 刚性误差 (MSE)
        pred_dist = torch.cdist(pred_x, pred_x, p=2)
        target_dist = torch.cdist(target_x, target_x, p=2)
        rigid_loss = self.mse(pred_dist, target_dist)
        
        # 不要用 v^2 (动能)，改用 v 的模长 (速率)
        # 不要用 MSE (平方惩罚)，改用 L1 (线性惩罚) 或 Huber Loss
        # 这样即使模型偶尔预测出大速度，Loss 也不会爆炸
        
        pred_speed = torch.norm(pred_v, dim=-1)   # [B, N]
        target_speed = torch.norm(target_v, dim=-1) # [B, N]
        
        # 使用 L1 Loss 约束速率一致性
        energy_loss = self.l1(pred_speed, target_speed)
        
        total_loss = traj_loss + \
                     self.lambda_rigid * rigid_loss + \
                     self.lambda_energy * energy_loss
        
        return total_loss, traj_loss, rigid_loss, energy_loss

import os
import torch
from torch.utils.data import Dataset

class SapienSequenceDataset(Dataset):
    """
    加载序列化数据，支持 H-Step Rollout 训练。
    参数:
        sub_seq_len (int): 训练时截取的子序列长度 (例如 5+1 帧)
    """
    def __init__(self, data_path: str, sub_seq_len: int = 6) -> None:
        super().__init__()
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        print(f"Loading sequence dataset: {data_path} ...")
        # 使用 weights_only=True 提高安全性
        self.data = torch.load(data_path, weights_only=True)

        self.x = self.data["x"]       # [N_Traj, Total_Len, N, 3]
        self.v = self.data["v"]       # [N_Traj, Total_Len, N, 3]
        self.explicit = self.data["explicit"] # [N_Traj, Total_Len, 1, 3]
        self.context = self.data["context"]   # [N_Traj, Total_Len, 3]
        
        # 核心修复：这里定义参数名为 sub_seq_len，与 train.py 保持一致
        self.sub_seq_len = sub_seq_len
        
        self.num_traj = self.x.shape[0]
        self.traj_len = self.x.shape[1]
        
        # 计算全局统计量 (用于归一化)
        # 注意：这里我们使用 (0, 1, 2) 聚合 Traj, Time, Points 维度
        self.pos_mean = self.x.mean(dim=(0, 1, 2), keepdim=True) # [1, 1, 1, 3]
        self.pos_std = self.x.std(dim=(0, 1, 2), keepdim=True).clamp(min=1e-6)
        self.vel_mean = self.v.mean(dim=(0, 1, 2), keepdim=True)
        self.vel_std = self.v.std(dim=(0, 1, 2), keepdim=True).clamp(min=1e-6)
        
        # 调整 shape 以便广播: [1, 1, 1, 3] -> [1, 1, 3]
        # 这样在 __getitem__ 里可以直接和 [Seq, N, 3] 进行运算
        self.pos_mean = self.pos_mean.view(1, 1, 3)
        self.pos_std = self.pos_std.view(1, 1, 3)
        self.vel_mean = self.vel_mean.view(1, 1, 3)
        self.vel_std = self.vel_std.view(1, 1, 3)

        print(f"Stats loaded. Pos Std: {self.pos_std.mean():.4f}, Vel Std: {self.vel_std.mean():.4f}")

    def __len__(self) -> int:
        # 我们可以从每条轨迹中采样多少个片段
        # 例如轨迹长 20，我们需要 6 帧，那么有 20 - 6 = 14 个起点
        samples_per_traj = self.traj_len - self.sub_seq_len
        # 如果轨迹太短，无法采样
        if samples_per_traj <= 0:
             raise ValueError(f"Trajectory length ({self.traj_len}) is shorter than required sub_seq_len ({self.sub_seq_len})")
             
        return self.num_traj * samples_per_traj

    def __getitem__(self, idx: int):
        # 将线性 idx 映射到 (traj_idx, start_frame)
        samples_per_traj = self.traj_len - self.sub_seq_len
        traj_idx = idx // samples_per_traj
        start_t = idx % samples_per_traj
        
        end_t = start_t + self.sub_seq_len
        
        # 提取序列片段
        x_seq = self.x[traj_idx, start_t:end_t] # [Seq, N, 3]
        v_seq = self.v[traj_idx, start_t:end_t]
        explicit_seq = self.explicit[traj_idx, start_t:end_t]
        context_seq = self.context[traj_idx, start_t:end_t]
        
        # 归一化 (Normalize)
        x_norm = (x_seq - self.pos_mean) / self.pos_std
        v_norm = (v_seq - self.vel_mean) / self.vel_std
        
        # 返回整个序列
        return x_norm.float(), v_norm.float(), explicit_seq.float(), context_seq.float()

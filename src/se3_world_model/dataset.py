import os
import torch
from torch.utils.data import Dataset

class SapienSequenceDataset(Dataset):
    """
    加载序列数据，用于多步训练。
    Returns: 
        x_seq: [Seq_Len, N, 3]
        v_seq: [Seq_Len, N, 3]
        explicit: [Seq_Len, 1, 3]
        context: [Seq_Len, 3]
    """
    def __init__(self, data_path: str, sub_seq_len: int = 5) -> None:
        super().__init__()
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found.")
            return

        print(f"Loading sequence dataset: {data_path} ...")
        # weights_only=True 提高安全性
        self.data = torch.load(data_path, weights_only=True)
        
        self.x = self.data["x"]       # [N_traj, Full_Seq, N, 3]
        self.v = self.data["v"]
        self.explicit = self.data["explicit"]
        self.context = self.data["context"]
        
        self.sub_seq_len = sub_seq_len
        self.num_traj = self.x.shape[0]
        self.full_seq_len = self.x.shape[1]
        
        # === 修复：确保统计量维度正确 ===
        # 原始维度 [N_traj, Full_Seq, N, 3] -> mean -> [1, 1, 1, 3]
        # 我们需要将其 view 为 [1, 1, 3] 以便与 [S, N, 3] 正确广播
        self.pos_mean = self.x.mean(dim=(0,1,2)).view(1, 1, 3)
        self.pos_std = self.x.std(dim=(0,1,2)).clamp(min=1e-6).view(1, 1, 3)
        
        self.vel_mean = self.v.mean(dim=(0,1,2)).view(1, 1, 3)
        self.vel_std = self.v.std(dim=(0,1,2)).clamp(min=1e-6).view(1, 1, 3)
        
        print(f"Stats - Pos Std: {self.pos_std.mean():.4f}, Vel Std: {self.vel_std.mean():.4f}")

    def __len__(self):
        # 简单起见，每条轨迹作为一个样本来源
        return self.num_traj

    def __getitem__(self, idx):
        # 随机选择一个起始点
        max_start = self.full_seq_len - self.sub_seq_len
        t_start = torch.randint(0, max_start + 1, (1,)).item()
        
        # 提取切片 [Sub_Seq, N, 3]
        x_seq = self.x[idx, t_start : t_start + self.sub_seq_len]
        v_seq = self.v[idx, t_start : t_start + self.sub_seq_len]
        explicit = self.explicit[idx, t_start : t_start + self.sub_seq_len]
        context = self.context[idx, t_start : t_start + self.sub_seq_len]
        
        # 归一化
        # [S, N, 3] - [1, 1, 3] -> [S, N, 3] (保持3维)
        x_norm = (x_seq - self.pos_mean) / self.pos_std
        v_norm = (v_seq - self.vel_mean) / self.vel_std
        
        return x_norm.float(), v_norm.float(), explicit.float(), context.float()

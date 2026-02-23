import os
import torch
from torch.utils.data import Dataset

class SapienSequenceDataset(Dataset):
    def __init__(self, data_path: str, sub_seq_len: int = 6) -> None:
        super().__init__()
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        print(f"Loading sequence dataset: {data_path} ...")
        self.data = torch.load(data_path, weights_only=True)

        self.x = self.data["x"]       
        self.v = self.data["v"]       
        self.explicit = self.data["explicit"] 
        self.context = self.data["context"]   
        
        self.sub_seq_len = sub_seq_len
        self.num_traj = self.x.shape[0]
        self.traj_len = self.x.shape[1]
        
        self.pos_mean = self.x.mean(dim=(0, 1, 2), keepdim=True) 
        self.vel_mean = self.v.mean(dim=(0, 1, 2), keepdim=True)
        
        # === 终极物理修复：统一欧氏度量空间 ===
        # 强制位置和速度使用相同的缩放因子，保持刚体旋转的数学自洽
        shared_std = self.v.std(dim=(0, 1, 2), keepdim=True).clamp(min=1e-6)
        self.pos_std = shared_std
        self.vel_std = shared_std
        
        self.pos_mean = self.pos_mean.view(1, 1, 3)
        self.pos_std = self.pos_std.view(1, 1, 3)
        self.vel_mean = self.vel_mean.view(1, 1, 3)
        self.vel_std = self.vel_std.view(1, 1, 3)

        print(f"Stats loaded. Shared Std: {self.pos_std.mean():.4f}")

    def __len__(self) -> int:
        samples_per_traj = self.traj_len - self.sub_seq_len
        if samples_per_traj <= 0:
             raise ValueError("Trajectory length too short")
        return self.num_traj * samples_per_traj

    def __getitem__(self, idx: int):
        samples_per_traj = self.traj_len - self.sub_seq_len
        traj_idx = idx // samples_per_traj
        start_t = idx % samples_per_traj
        end_t = start_t + self.sub_seq_len
        
        x_seq = self.x[traj_idx, start_t:end_t] 
        v_seq = self.v[traj_idx, start_t:end_t]
        explicit_seq = self.explicit[traj_idx, start_t:end_t]
        context_seq = self.context[traj_idx, start_t:end_t]
        
        x_norm = (x_seq - self.pos_mean) / self.pos_std
        v_norm = (v_seq - self.vel_mean) / self.vel_std
        
        return x_norm.float(), v_norm.float(), explicit_seq.float(), context_seq.float()

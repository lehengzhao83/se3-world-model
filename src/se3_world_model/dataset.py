import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class SapienSequenceDataset(Dataset):
    def __init__(self, data_path: str, sub_seq_len: int = 6) -> None:
        super().__init__()
        self.data_path = data_path
        self.sub_seq_len = sub_seq_len
        
        with h5py.File(data_path, 'r') as f:
            self.num_traj = f['x'].shape[0]
            self.traj_len = f['x'].shape[1]
            
            self.pos_mean = torch.tensor(np.array(f.attrs['pos_mean'])).float().view(1, 1, 3)
            self.vel_mean = torch.tensor(np.array(f.attrs['vel_mean'])).float().view(1, 1, 3)
            self.force_mean = torch.tensor(np.array(f.attrs['force_mean'])).float().view(1, 1, 3)
            
            shared_vel_std = torch.tensor(np.array(f.attrs['vel_std'])).float().clamp(min=1e-6)
            self.pos_std = torch.tensor([1.0, 1.0, 1.0]).float().view(1, 1, 3)
            self.vel_std = shared_vel_std.view(1, 1, 1).expand(1, 1, 3)
            
            force_std = torch.tensor(np.array(f.attrs['force_std'])).float().clamp(min=1e-6)
            self.force_std = force_std.view(1, 1, 1).expand(1, 1, 3)

    def __len__(self) -> int:
        samples_per_traj = self.traj_len - self.sub_seq_len
        return self.num_traj * samples_per_traj

    def __getitem__(self, idx: int):
        with h5py.File(self.data_path, 'r') as f:
            samples_per_traj = self.traj_len - self.sub_seq_len
            traj_idx = idx // samples_per_traj     
            start_t = idx % samples_per_traj       
            end_t = start_t + self.sub_seq_len     
            
            x_seq = torch.from_numpy(f['x'][traj_idx, start_t:end_t])
            v_seq = torch.from_numpy(f['v'][traj_idx, start_t:end_t])
            f_seq = torch.from_numpy(f['force'][traj_idx, start_t:end_t])
            explicit_seq = torch.from_numpy(f['explicit'][traj_idx, start_t:end_t])
            context_seq = torch.from_numpy(f['context'][traj_idx, start_t:end_t])
        
        x_norm = (x_seq - self.pos_mean) / self.pos_std
        # 【修复等变性】：绝对不能减去全局固定常数 vel_mean，只做标量缩放
        v_norm = v_seq / self.vel_std
        f_norm = (f_seq - self.force_mean) / self.force_std
        
        return x_norm.float(), v_norm.float(), f_norm.float(), explicit_seq.float(), context_seq.float()

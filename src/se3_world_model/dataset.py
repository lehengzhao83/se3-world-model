import os
import torch
from torch.utils.data import Dataset

class SapienDataset(Dataset):
    """
    Input: Normalized Pos (x), Normalized Vel (v), Explicit, Context
    Target: Normalized Next Vel (v_next) -- NOT Position!
    """
    def __init__(self, data_path: str) -> None:
        super().__init__()
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        print(f"Loading dataset: {data_path} ...")
        # weights_only=True for security
        self.data = torch.load(data_path, weights_only=True)

        self.x_t = self.data["x_t"]
        self.v_t = self.data["v_t"]
        self.explicit = self.data["explicit"]
        self.context = self.data["context"]
        self.x_next = self.data["x_next"]

        # 1. Position Stats (Global)
        all_pos = torch.cat([self.x_t, self.x_next], dim=0)
        self.pos_mean = all_pos.mean(dim=(0, 1), keepdim=True)
        self.pos_std = all_pos.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)

        # 2. Velocity Stats (Local - Critical!)
        # Calculate real v_next from data to ensure consistency
        v_next_real = self.x_next - self.x_t
        all_vel = torch.cat([self.v_t, v_next_real], dim=0)
        
        # Velocity is a vector, mean should be close to 0, but std is important
        self.vel_mean = all_vel.mean(dim=(0, 1), keepdim=True)
        self.vel_std = all_vel.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)

        print(f"Stats - Pos Std: {self.pos_std.mean().item():.4f} | Vel Std: {self.vel_std.mean().item():.4f}")
        # Explain: If Pos Std >> Vel Std, previous model failed because delta was too small.

    def __len__(self) -> int:
        return len(self.x_t)

    def __getitem__(self, idx: int):
        # Raw Data
        x = self.x_t[idx]
        v = self.v_t[idx]
        x_next = self.x_next[idx]

        # Normalize Position using Pos Stats
        x_norm = (x - self.pos_mean[0]) / self.pos_std[0]
        
        # Normalize Velocity using Vel Stats (Amplification)
        v_norm = (v - self.vel_mean[0]) / self.vel_std[0]
        
        # Target: Normalized Next Velocity
        v_next_real = x_next - x
        target_v_norm = (v_next_real - self.vel_mean[0]) / self.vel_std[0]
        
        return (
            x_norm.float(), 
            v_norm.float(), 
            self.explicit[idx].float(), 
            self.context[idx].float(), 
            target_v_norm.float() # Target is Velocity!
        )

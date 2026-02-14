import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from se3_world_model.dataset import SapienSequenceDataset
from se3_world_model.model import SE3WorldModel

def setup_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size
    return 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized(): dist.destroy_process_group()

def train(epochs: int = 50, batch_size: int = 32, lr: float = 1e-3, save_dir: str = "checkpoints"):
    local_rank, global_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # 1. 使用序列 Dataset
    # 训练时每次看 5 步 (H=5)
    rollout_steps = 5
    dataset = SapienSequenceDataset("data/sapien_train_seq.pt", sub_seq_len=rollout_steps+1)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    # 获取归一化比例，用于在 Rollout 中正确积分位置
    # ratio = Vel_Std / Pos_Std
    scale_ratio = (dataset.vel_std / dataset.pos_std).to(device)

    model = SE3WorldModel(num_points=64, latent_dim=64, num_global_vectors=1, context_dim=3).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if global_rank == 0:
        print(f"=== 启动 H-Step Rollout Training (H={rollout_steps}) ===")
        os.makedirs(save_dir, exist_ok=True)

    model.train()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        total_loss = 0.0
        
        # Data: [B, H+1, N, 3]
        for batch_idx, (x_seq, v_seq, explicit_seq, context_seq) in enumerate(dataloader):
            x_seq = x_seq.to(device, non_blocking=True)
            v_seq = v_seq.to(device, non_blocking=True)
            explicit_seq = explicit_seq.to(device, non_blocking=True)
            context_seq = context_seq.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # 初始化状态 (t=0)
            curr_x = x_seq[:, 0]
            curr_v = v_seq[:, 0]
            
            loss_accum = 0.0
            
            # === 核心：自回归 Rollout 循环 ===
            for t in range(rollout_steps):
                # 1. 预测下一步速度
                # explicit/context 取当前时刻 t
                pred_v_next, _ = model(curr_x, curr_v, explicit_seq[:, t], context_seq[:, t])
                
                # 2. 计算当前步的 Loss
                # Target 是序列中真实的 t+1 时刻速度
                target_v_next = v_seq[:, t+1]
                loss_accum += criterion(pred_v_next, target_v_next)
                
                # 3. 状态更新 (Differentiable Physics Integration)
                # 这一步让梯度可以通过时间反向传播 (BPTT)
                
                # 更新位置: x_{t+1} = x_t + v_{t+1}
                # 注意：这是在 Normalized 空间更新，需要乘以 scale_ratio 才能符合物理
                # x_norm_new = x_norm_old + v_norm_new * (vel_std / pos_std)
                curr_x = curr_x + pred_v_next * scale_ratio
                
                # 更新速度: v_{t+1} = pred_v_next
                curr_v = pred_v_next
            
            # 平均 Loss
            loss = loss_accum / rollout_steps
            
            loss.backward()
            # 梯度裁剪防止爆炸 (BPTT 常见问题)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()

            if global_rank == 0 and batch_idx % 10 == 0:
                print(f"[GPU 0] Epoch {epoch+1} Step {batch_idx} Loss: {loss.item():.6f}")

        if global_rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"=== Epoch {epoch+1} Avg Loss: {avg_loss:.6f} ===")
            torch.save(model.module.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))

    cleanup_distributed()

if __name__ == "__main__":
    train(epochs=50, batch_size=32)

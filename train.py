import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
# 引用新的 Dataset
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

    # 1. 加载序列数据
    # seq_len = 6 (输入1帧 + 预测5帧)
    rollout_steps = 5 
    dataset = SapienSequenceDataset("data/sapien_train_seq.pt", seq_len=rollout_steps+1)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    # 获取缩放比例：用于在 Rollout 时把预测的速度加回到位置上
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
        
        # x_seq: [B, Seq, N, 3]
        for batch_idx, (x_seq, v_seq, explicit_seq, context_seq) in enumerate(dataloader):
            x_seq = x_seq.to(device, non_blocking=True)
            v_seq = v_seq.to(device, non_blocking=True)
            explicit_seq = explicit_seq.to(device, non_blocking=True)
            context_seq = context_seq.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # 初始状态 (t=0)
            curr_x = x_seq[:, 0]
            curr_v = v_seq[:, 0]
            
            rollout_loss = 0.0
            
            # === 核心：Rollout 循环 ===
            for t in range(rollout_steps):
                # 1. 预测下一帧速度
                # 使用当前预测出的 curr_x 和 curr_v
                pred_v_next, _ = model(curr_x, curr_v, explicit_seq[:, t], context_seq[:, t])
                
                # 2. 计算速度 Loss
                target_v_next = v_seq[:, t+1]
                step_loss = criterion(pred_v_next, target_v_next)
                rollout_loss += step_loss
                
                # 3. 状态更新 (积分)
                # x_{t+1} = x_t + v_{t+1} * ratio
                # 这一步至关重要：它让误差传播到了 curr_x，
                # 如果这一步预测偏了，下一步的输入 curr_x 就会偏，导致下一步 Loss 变大。
                curr_x = curr_x + pred_v_next * scale_ratio
                curr_v = pred_v_next # 更新速度用于下一步输入
            
            # 平均多步 Loss
            final_loss = rollout_loss / rollout_steps
            
            final_loss.backward()
            # 梯度裁剪防止 Rollout 导致的梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += final_loss.item()

            if global_rank == 0 and batch_idx % 10 == 0:
                print(f"[GPU 0] Epoch {epoch+1} Step {batch_idx} Rollout Loss: {final_loss.item():.6f}")

        if global_rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"=== Epoch {epoch+1} Avg Loss: {avg_loss:.6f} ===")
            torch.save(model.module.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))

    cleanup_distributed()

if __name__ == "__main__":
    train(epochs=50, batch_size=32)

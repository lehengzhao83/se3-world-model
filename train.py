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
from se3_world_model.dataset import SapienSequenceDataset
from se3_world_model.model import SE3WorldModel
# 1. 引入你写的 Loss
from se3_world_model.loss import GeometricConsistencyLoss

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

    # === 配置：课程学习参数 ===
    # 你的数据有 20 帧，我们可以最大训练 10-15 步的 Rollout
    MAX_ROLLOUT_STEPS = 15
    START_ROLLOUT_STEPS = 5
    # 每隔多少个 Epoch 增加一步 Rollout
    INCREASE_ROLLOUT_EVERY = 5 

    # 1. 加载足够长的数据序列
    # 我们一次性加载 MAX_ROLLOUT + 1 帧，训练时根据当前课程动态切片
    dataset = SapienSequenceDataset("data/sapien_train_seq.pt", sub_seq_len=MAX_ROLLOUT_STEPS+1)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    # 获取缩放比例：用于积分 x_next = x_curr + v_pred * ratio
    scale_ratio = (dataset.vel_std / dataset.pos_std).to(device)

    model = SE3WorldModel(num_points=64, latent_dim=64, num_global_vectors=1, context_dim=3).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 2. 定义双重 Loss
    # 速度 Loss (MSE)
    criterion_vel = nn.MSELoss()
    # 位置 Loss (MSE + Rigidity) - 这就是你一直没用上的杀手锏
    criterion_pos = GeometricConsistencyLoss(lambda_rigid=0.5).to(device)

    if global_rank == 0:
        print(f"=== 启动 Curriculum Rollout Training ===")
        print(f"Max Horizon: {MAX_ROLLOUT_STEPS}, Strategy: +1 step every {INCREASE_ROLLOUT_EVERY} epochs")
        os.makedirs(save_dir, exist_ok=True)

    model.train()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        
        # === 核心：动态调整 Rollout 步数 (Curriculum Learning) ===
        # 随着训练进行，步数从 5 慢慢涨到 15
        current_H = min(MAX_ROLLOUT_STEPS, START_ROLLOUT_STEPS + epoch // INCREASE_ROLLOUT_EVERY)
        
        total_loss = 0.0
        total_rigid = 0.0
        
        # x_seq shape: [B, MAX_H+1, N, 3]
        for batch_idx, (x_seq, v_seq, explicit_seq, context_seq) in enumerate(dataloader):
            # Move to GPU
            x_seq = x_seq.to(device, non_blocking=True)
            v_seq = v_seq.to(device, non_blocking=True)
            explicit_seq = explicit_seq.to(device, non_blocking=True)
            context_seq = context_seq.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # 初始化推演状态
            curr_x = x_seq[:, 0]
            curr_v = v_seq[:, 0]
            
            batch_loss = 0.0
            
            # === Rollout 循环 ===
            for t in range(current_H):
                # Ground Truth for this step
                target_v = v_seq[:, t+1]
                target_x = x_seq[:, t+1] # 我们现在也要看位置了！
                
                # Context
                curr_expl = explicit_seq[:, t]
                curr_ctx = context_seq[:, t]
                
                # 1. 预测速度
                pred_v_norm, _ = model(curr_x, curr_v, curr_expl, curr_ctx)
                
                # 2. 积分得到位置 (Differentiable Integration)
                # 这一步非常关键：它把 pred_v_norm 的梯度和 curr_x 挂钩了
                # 如果 pred_v 预测偏了，curr_x 就会偏，进而在 Pos Loss 中受到惩罚
                next_x = curr_x + pred_v_norm * scale_ratio
                
                # 3. 计算混合 Loss
                # A. 速度要准 (动力学约束)
                loss_v = criterion_vel(pred_v_norm, target_v)
                
                # B. 位置要准 + 形状要硬 (几何约束)
                # 使用你定义的 GeometricConsistencyLoss
                loss_x_total, loss_traj, loss_rigid = criterion_pos(next_x, target_x)
                
                # C. 总 Loss (可以加权，这里暂定 1:1)
                # 位置 Loss 通常数值较大，可以给个系数 0.1 或 1.0，视量级而定
                step_loss = loss_v + loss_x_total
                
                batch_loss += step_loss
                
                # 记录一下刚性损失用于监控
                total_rigid += loss_rigid.item()
                
                # 更新状态用于下一步
                curr_x = next_x
                curr_v = pred_v_norm # 自回归：吃自己的预测
            
            # 平均 Loss 并反向传播
            final_loss = batch_loss / current_H
            final_loss.backward()
            
            # 梯度裁剪 (长程 RNN/Rollout 必备)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += final_loss.item()

            if global_rank == 0 and batch_idx % 10 == 0:
                print(f"[GPU 0] Epoch {epoch+1} (H={current_H}) Step {batch_idx} "
                      f"Loss: {final_loss.item():.4f} | Rigid: {loss_rigid.item():.4f}")

        if global_rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"=== Epoch {epoch+1} Avg Loss: {avg_loss:.6f} ===")
            torch.save(model.module.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))

    cleanup_distributed()

if __name__ == "__main__":
    train(epochs=50, batch_size=32)

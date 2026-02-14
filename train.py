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

def train(epochs: int = 100, batch_size: int = 32, lr: float = 1e-3, save_dir: str = "checkpoints"):
    local_rank, global_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # === 配置：课程学习参数 ===
    MAX_ROLLOUT_STEPS = 15
    START_ROLLOUT_STEPS = 5
    # 稍微放慢课程难度增加的速度，让模型基础打得更牢
    INCREASE_ROLLOUT_EVERY = 8 

    # 1. 加载数据
    dataset = SapienSequenceDataset("data/sapien_train_seq.pt", sub_seq_len=MAX_ROLLOUT_STEPS+1)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    # 获取缩放比例
    scale_ratio = (dataset.vel_std / dataset.pos_std).to(device)

    # === 改进点 1: 增加模型容量 ===
    # latent_dim 从 64 提升到 128，增强对复杂物理规律的记忆能力
    model = SE3WorldModel(num_points=64, latent_dim=128, num_global_vectors=1, context_dim=3).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # === 改进点 2: 调整 Loss 权重 ===
    criterion_vel = nn.MSELoss()
    
    # 刚性惩罚 (lambda_rigid) 从 0.5 提升到 2.0 -> "像钢板一样硬"
    criterion_pos = GeometricConsistencyLoss(lambda_rigid=2.0).to(device)

    # 位置误差权重 -> "最终到了哪里比中间跑得快不快更重要"
    LOSS_WEIGHT_POS = 5.0 

    if global_rank == 0:
        print(f"=== 启动 Enhanced Curriculum Rollout Training ===")
        print(f"Configs: Latent=128, Rigid=2.0, PosWeight={LOSS_WEIGHT_POS}, MaxSteps={MAX_ROLLOUT_STEPS}")
        os.makedirs(save_dir, exist_ok=True)

    model.train()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        
        # 动态调整 Rollout 步数
        current_H = min(MAX_ROLLOUT_STEPS, START_ROLLOUT_STEPS + epoch // INCREASE_ROLLOUT_EVERY)
        
        total_loss = 0.0
        total_rigid = 0.0
        
        for batch_idx, (x_seq, v_seq, explicit_seq, context_seq) in enumerate(dataloader):
            x_seq = x_seq.to(device, non_blocking=True)
            v_seq = v_seq.to(device, non_blocking=True)
            explicit_seq = explicit_seq.to(device, non_blocking=True)
            context_seq = context_seq.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            curr_x = x_seq[:, 0]
            curr_v = v_seq[:, 0]
            
            batch_loss = 0.0
            
            # Rollout 循环
            for t in range(current_H):
                target_v = v_seq[:, t+1]
                target_x = x_seq[:, t+1]
                
                curr_expl = explicit_seq[:, t]
                curr_ctx = context_seq[:, t]
                
                # 1. 预测速度
                pred_v_norm, _ = model(curr_x, curr_v, curr_expl, curr_ctx)
                
                # 2. 积分得到位置
                next_x = curr_x + pred_v_norm * scale_ratio
                
                # 3. 计算混合 Loss
                loss_v = criterion_vel(pred_v_norm, target_v)
                loss_x_total, _, loss_rigid = criterion_pos(next_x, target_x)
                
                # === 改进点 3: 强力纠偏 ===
                # 如果位置偏了，给予 5 倍的惩罚，迫使模型在下一帧修正回来
                step_loss = loss_v + (LOSS_WEIGHT_POS * loss_x_total)
                
                batch_loss += step_loss
                total_rigid += loss_rigid.item()
                
                # 更新状态
                curr_x = next_x
                curr_v = pred_v_norm 
            
            final_loss = batch_loss / current_H
            final_loss.backward()
            
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
    # 增加到 100 个 Epoch，给模型更多时间适应长程预测
    train(epochs=100, batch_size=32)

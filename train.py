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

# 确保 src 目录在 python path 中
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from se3_world_model.dataset import SapienSequenceDataset
from se3_world_model.model import SE3WorldModel
# 引用包含能量约束的新 Loss
from se3_world_model.loss import GeometricConsistencyLoss

def setup_distributed() -> tuple[int, int, int]:
    """初始化分布式训练环境"""
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

def train(
    epochs: int = 100, 
    batch_size: int = 32, 
    lr: float = 1e-3, 
    save_dir: str = "checkpoints"
):
    local_rank, global_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # === 配置：课程学习参数 ===
    # 最大 Rollout 步数
    MAX_ROLLOUT_STEPS = 15
    # 初始 Rollout 步数
    START_ROLLOUT_STEPS = 5
    # 每隔多少个 Epoch 增加一步 Rollout
    INCREASE_ROLLOUT_EVERY = 8 

    # 1. 加载数据
    # sub_seq_len = 预测步数 + 1 (输入帧)
    dataset = SapienSequenceDataset("data/sapien_train_seq.pt", sub_seq_len=MAX_ROLLOUT_STEPS+1)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    # 获取缩放比例：用于积分 x_next = x_curr + v_pred * ratio
    scale_ratio = (dataset.vel_std / dataset.pos_std).to(device)

    # 2. 模型初始化
    # 注意：latent_dim=128，与 make_video.py 保持一致
    model = SE3WorldModel(num_points=64, latent_dim=128, num_global_vectors=1, context_dim=3).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 3. Loss 初始化
    criterion_vel = nn.MSELoss()
    # 刚性=2.0 (强约束形状), 能量=0.1 (弱约束防止速度爆炸)
    criterion_pos = GeometricConsistencyLoss(lambda_rigid=2.0, lambda_energy=0.1).to(device)
    
    # 位置误差权重：位置漂移是致命的，给予高权重
    LOSS_WEIGHT_POS = 2.0 

    if global_rank == 0:
        print(f"=== 启动 Enhanced Curriculum Rollout Training ===")
        print(f"Configs: Latent=128, Rigid=2.0, Energy=0.1, PosWeight={LOSS_WEIGHT_POS}")
        os.makedirs(save_dir, exist_ok=True)

    model.train()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        
        # === 动态调整 Rollout 步数 (Curriculum Learning) ===
        current_H = min(MAX_ROLLOUT_STEPS, START_ROLLOUT_STEPS + epoch // INCREASE_ROLLOUT_EVERY)
        
        # === 计划采样概率 (Teacher Forcing Ratio) ===
        # 从 1.0 (全靠老师) 线性衰减到 0.0 (全靠自己)
        # 前 20 个 epoch 逐渐放手，让模型学会自我纠错
        teacher_forcing_ratio = max(0.0, 1.0 - epoch / 20.0)
        
        total_loss = 0.0
        total_rigid = 0.0
        total_energy = 0.0
        
        for batch_idx, (x_seq, v_seq, explicit_seq, context_seq) in enumerate(dataloader):
            x_seq = x_seq.to(device, non_blocking=True)
            v_seq = v_seq.to(device, non_blocking=True)
            explicit_seq = explicit_seq.to(device, non_blocking=True)
            context_seq = context_seq.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # 初始化状态
            curr_x = x_seq[:, 0]
            curr_v = v_seq[:, 0]
            
            batch_loss = 0.0
            
            # === Rollout 循环 ===
            for t in range(current_H):
                # 目标数据
                target_v = v_seq[:, t+1]
                target_x = x_seq[:, t+1]
                
                # 当前上下文
                curr_expl = explicit_seq[:, t]
                curr_ctx = context_seq[:, t]
                
                # 1. 预测速度
                pred_v_norm, _ = model(curr_x, curr_v, curr_expl, curr_ctx)
                
                # 2. 积分得到位置 (Differentiable Integration)
                # 这一步将 pred_v_norm 的梯度与 next_x 的位置误差挂钩
                next_x = curr_x + pred_v_norm * scale_ratio
                
                # === 3. 混合策略 (Scheduled Sampling) ===
                # 决定下一步的输入速度是用“真实的”还是“预测的”
                if torch.rand(1).item() < teacher_forcing_ratio:
                    # Teacher Forcing: 使用真实速度作为下一步的惯性基础
                    next_input_v = target_v 
                else:
                    # Autoregressive: 使用自己的预测 (吃自己的狗粮)
                    next_input_v = pred_v_norm
                
                # 4. 计算 Loss
                loss_v = criterion_vel(pred_v_norm, target_v)
                
                # 注意：GeometricConsistencyLoss 现在接收 4 个参数用于计算能量损失
                loss_x_total, _, loss_rigid, loss_energy = criterion_pos(next_x, target_x, pred_v_norm, target_v)
                
                # 综合 Loss
                step_loss = loss_v + (LOSS_WEIGHT_POS * loss_x_total)
                
                batch_loss += step_loss
                total_rigid += loss_rigid.item()
                total_energy += loss_energy.item()
                
                # 更新状态用于下一步
                curr_x = next_x
                curr_v = next_input_v # <--- 混合后的速度输入
            
            # 平均 Loss 并反向传播
            final_loss = batch_loss / current_H
            final_loss.backward()
            
            # 梯度裁剪 (防止 Rollout 导致的梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_loss += final_loss.item()

            if global_rank == 0 and batch_idx % 10 == 0:
                print(f"[GPU 0] Epoch {epoch+1} (H={current_H}, TF={teacher_forcing_ratio:.2f}) Step {batch_idx} "
                      f"Loss: {final_loss.item():.4f} | Rigid: {loss_rigid.item():.4f} | Energy: {loss_energy.item():.4f}")

        if global_rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"=== Epoch {epoch+1} Avg Loss: {avg_loss:.6f} ===")
            torch.save(model.module.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))

    cleanup_distributed()

if __name__ == "__main__":
    train(epochs=100, batch_size=32)

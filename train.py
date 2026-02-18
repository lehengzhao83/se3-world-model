import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from se3_world_model.dataset import SapienSequenceDataset
from se3_world_model.model import SE3WorldModel
from se3_world_model.loss import GeometricConsistencyLoss

def setup_distributed():
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

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    local_rank, global_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # === Logger Setup ===
    tb_writer = None
    if global_rank == 0:
        print(f"=== Training Started: {cfg.logger.run_name} ===")
        print(OmegaConf.to_yaml(cfg))
        os.makedirs(cfg.training.save_dir, exist_ok=True)
        if cfg.logger.use_tensorboard:
            tb_writer = SummaryWriter(log_dir=os.path.join("runs", cfg.logger.run_name))
        if cfg.logger.use_wandb:
            wandb.init(project=cfg.logger.project_name, name=cfg.logger.run_name, config=OmegaConf.to_container(cfg, resolve=True))

    # === Parameters ===
    history_len = cfg.model.get("history_len", 1)
    max_rollout = cfg.training.curriculum.max_rollout_steps
    noise_std = cfg.training.get("noise_std", 0.0) # 获取噪声参数

    # === Dataset ===
    # 确保 Dataset 长度足够: History + Rollout
    dataset = SapienSequenceDataset(
        data_path=cfg.dataset.train_path, 
        sub_seq_len=max_rollout + history_len
    )
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.training.batch_size, 
        sampler=sampler, 
        num_workers=cfg.training.num_workers, 
        pin_memory=True
    )

    scale_ratio = (dataset.vel_std / dataset.pos_std).to(device)

    # === Model ===
    model = SE3WorldModel(
        num_points=cfg.model.num_points, 
        latent_dim=cfg.model.latent_dim, 
        num_global_vectors=cfg.model.num_global_vectors, 
        context_dim=cfg.model.context_dim,
        history_len=history_len
    ).to(device)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    
    criterion_vel = nn.MSELoss()
    criterion_pos = GeometricConsistencyLoss(
        lambda_rigid=cfg.training.loss_weights.rigid, 
        lambda_energy=cfg.training.loss_weights.energy
    ).to(device)

    model.train()

    # === Training Loop ===
    global_step = 0
    
    for epoch in range(cfg.training.epochs):
        sampler.set_epoch(epoch)
        
        # Curriculum Schedule
        curr_config = cfg.training.curriculum
        current_rollout_steps = min(
            curr_config.max_rollout_steps, 
            curr_config.start_rollout_steps + epoch // curr_config.increase_every_epochs
        )
        
        # Teacher Forcing Schedule
        tf_decay_steps = float(curr_config.teacher_forcing_decay_epochs)
        teacher_forcing_ratio = max(0.0, 1.0 - epoch / tf_decay_steps)
        
        epoch_loss = 0.0
        
        for batch_idx, (x_seq, v_seq, explicit_seq, context_seq) in enumerate(dataloader):
            x_seq = x_seq.to(device, non_blocking=True)
            v_seq = v_seq.to(device, non_blocking=True)
            explicit_seq = explicit_seq.to(device, non_blocking=True)
            context_seq = context_seq.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # 初始化 History Buffer
            curr_x_hist = x_seq[:, 0:history_len] # [B, H, N, 3]
            curr_v_hist = v_seq[:, 0:history_len]
            
            batch_loss = 0.0
            total_rigid_loss = 0.0
            
            # Rollout
            for t in range(current_rollout_steps):
                target_idx = history_len + t
                target_v = v_seq[:, target_idx]
                target_x = x_seq[:, target_idx]
                
                # Context (取 Input 序列最后一帧对应的时间点)
                ctx_idx = target_idx - 1
                curr_expl = explicit_seq[:, ctx_idx]
                curr_ctx = context_seq[:, ctx_idx]
                
                # === 关键：噪声注入 (Noise Injection) ===
                # 我们 Clone 一份用于输入，防止污染用于 Update 的真实历史
                if model.training and noise_std > 0:
                    input_x_hist = curr_x_hist.clone() + torch.randn_like(curr_x_hist) * noise_std
                    input_v_hist = curr_v_hist.clone() + torch.randn_like(curr_v_hist) * noise_std
                else:
                    input_x_hist = curr_x_hist
                    input_v_hist = curr_v_hist

                # Forward
                pred_v_norm, _ = model(input_x_hist, input_v_hist, curr_expl, curr_ctx)
                
                # 积分: 使用 History 中最新的一帧 (无噪声) 进行物理更新
                last_x = curr_x_hist[:, -1] 
                next_x = last_x + pred_v_norm * scale_ratio
                
                # Loss
                loss_v = criterion_vel(pred_v_norm, target_v)
                loss_x_total, _, loss_rigid, loss_energy = criterion_pos(next_x, target_x, pred_v_norm, target_v)
                
                step_loss = loss_v + (cfg.training.loss_weights.pos * loss_x_total)
                batch_loss += step_loss
                total_rigid_loss += loss_rigid
                
                # Update Buffer (Sliding Window)
                # 决定下一帧输入是真实值还是预测值
                use_ground_truth = (torch.rand(1).item() < teacher_forcing_ratio)
                
                if use_ground_truth:
                    next_v_in = target_v
                    next_x_in = target_x
                else:
                    next_v_in = pred_v_norm.detach() # Detach防止梯度回传过远
                    next_x_in = next_x.detach()

                curr_x_hist = torch.cat([curr_x_hist[:, 1:], next_x_in.unsqueeze(1)], dim=1)
                curr_v_hist = torch.cat([curr_v_hist[:, 1:], next_v_in.unsqueeze(1)], dim=1)

            final_loss = batch_loss / current_rollout_steps
            final_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.grad_clip)
            optimizer.step()
            
            epoch_loss += final_loss.item()
            
            # Step Logging
            if global_rank == 0 and global_step % 10 == 0:
                if cfg.logger.use_wandb:
                    wandb.log({
                        "train_loss": final_loss.item(),
                        "rollout_steps": current_rollout_steps,
                        "tf_ratio": teacher_forcing_ratio
                    }, step=global_step)
            global_step += 1

        # Epoch Logging
        avg_epoch_loss = epoch_loss / len(dataloader)
        if global_rank == 0:
            print(f"Epoch {epoch+1} done. Avg Loss: {avg_epoch_loss:.6f}")
            save_path = os.path.join(cfg.training.save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.module.state_dict(), save_path)

    if global_rank == 0:
        if cfg.logger.use_wandb: wandb.finish()
    cleanup_distributed()

if __name__ == "__main__":
    main()

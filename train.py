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

# === Hydra 入口装饰器 ===
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    local_rank, global_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # === 1. 初始化 Logger (仅在主进程) ===
    tb_writer = None
    if global_rank == 0:
        print(f"=== Training Started: {cfg.logger.run_name} ===")
        print(OmegaConf.to_yaml(cfg))
        
        # 创建检查点目录
        os.makedirs(cfg.training.save_dir, exist_ok=True)

        # TensorBoard
        if cfg.logger.use_tensorboard:
            tb_writer = SummaryWriter(log_dir=os.path.join("runs", cfg.logger.run_name))

        # WandB
        if cfg.logger.use_wandb:
            wandb.init(
                project=cfg.logger.project_name,
                name=cfg.logger.run_name,
                config=OmegaConf.to_container(cfg, resolve=True)
            )

    # === 2. 加载数据 ===
    # 从 cfg 读取参数
    max_steps = cfg.training.curriculum.max_rollout_steps
    train_path = cfg.dataset.train_path
    
    dataset = SapienSequenceDataset(
        data_path=train_path, 
        sub_seq_len=max_steps + 1
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

    # === 3. 模型初始化 ===
    model = SE3WorldModel(
        num_points=cfg.model.num_points, 
        latent_dim=cfg.model.latent_dim, 
        num_global_vectors=cfg.model.num_global_vectors, 
        context_dim=cfg.model.context_dim
    ).to(device)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    
    # Loss 配置
    criterion_vel = nn.MSELoss()
    criterion_pos = GeometricConsistencyLoss(
        lambda_rigid=cfg.training.loss_weights.rigid, 
        lambda_energy=cfg.training.loss_weights.energy
    ).to(device)

    model.train()

    # === 4. 训练循环 ===
    global_step = 0
    
    for epoch in range(cfg.training.epochs):
        sampler.set_epoch(epoch)
        
        # Curriculum Learning: 动态计算当前 Rollout 步数
        curr_config = cfg.training.curriculum
        current_H = min(
            curr_config.max_rollout_steps, 
            curr_config.start_rollout_steps + epoch // curr_config.increase_every_epochs
        )
        
        # Teacher Forcing Ratio Decay
        tf_decay_steps = curr_config.teacher_forcing_decay_epochs
        teacher_forcing_ratio = max(0.0, 1.0 - epoch / float(tf_decay_steps))
        
        epoch_loss = 0.0
        
        for batch_idx, (x_seq, v_seq, explicit_seq, context_seq) in enumerate(dataloader):
            x_seq = x_seq.to(device, non_blocking=True)
            v_seq = v_seq.to(device, non_blocking=True)
            explicit_seq = explicit_seq.to(device, non_blocking=True)
            context_seq = context_seq.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            curr_x = x_seq[:, 0]
            curr_v = v_seq[:, 0]
            
            batch_loss = 0.0
            total_rigid_loss = 0.0
            total_energy_loss = 0.0
            
            # Rollout Loop
            for t in range(current_H):
                target_v = v_seq[:, t+1]
                target_x = x_seq[:, t+1]
                curr_expl = explicit_seq[:, t]
                curr_ctx = context_seq[:, t]
                
                pred_v_norm, _ = model(curr_x, curr_v, curr_expl, curr_ctx)
                next_x = curr_x + pred_v_norm * scale_ratio
                
                # Scheduled Sampling
                if torch.rand(1).item() < teacher_forcing_ratio:
                    next_input_v = target_v 
                else:
                    next_input_v = pred_v_norm
                
                # Loss Calculation
                loss_v = criterion_vel(pred_v_norm, target_v)
                loss_x_total, _, loss_rigid, loss_energy = criterion_pos(next_x, target_x, pred_v_norm, target_v)
                
                step_loss = loss_v + (cfg.training.loss_weights.pos * loss_x_total)
                
                batch_loss += step_loss
                total_rigid_loss += loss_rigid
                total_energy_loss += loss_energy
                
                curr_x = next_x
                curr_v = next_input_v 

            final_loss = batch_loss / current_H
            final_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.grad_clip)
            optimizer.step()
            
            epoch_loss += final_loss.item()
            
            # === 5. 实时记录 (Step Level) ===
            if global_rank == 0 and global_step % 10 == 0:
                # 记录到 TensorBoard
                if tb_writer:
                    tb_writer.add_scalar("Train/Loss", final_loss.item(), global_step)
                    tb_writer.add_scalar("Train/RigidLoss", total_rigid_loss.item() / current_H, global_step)
                    tb_writer.add_scalar("Train/EnergyLoss", total_energy_loss.item() / current_H, global_step)
                
                # 记录到 WandB
                if cfg.logger.use_wandb:
                    wandb.log({
                        "train_loss": final_loss.item(),
                        "rigid_loss": total_rigid_loss.item() / current_H,
                        "rollout_steps": current_H,
                        "teacher_forcing": teacher_forcing_ratio
                    }, step=global_step)
                    
            global_step += 1

        # === 6. Epoch 级别记录与保存 ===
        avg_epoch_loss = epoch_loss / len(dataloader)
        if global_rank == 0:
            print(f"Epoch {epoch+1} done. Avg Loss: {avg_epoch_loss:.6f}")
            
            if tb_writer:
                tb_writer.add_scalar("Train/EpochLoss", avg_epoch_loss, epoch)
            
            # 保存模型
            save_path = os.path.join(cfg.training.save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.module.state_dict(), save_path)

    # 清理
    if global_rank == 0:
        if tb_writer: tb_writer.close()
        if cfg.logger.use_wandb: wandb.finish()
    
    cleanup_distributed()

if __name__ == "__main__":
    main()

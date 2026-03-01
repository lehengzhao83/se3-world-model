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

from torch import amp 

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
    if dist.is_initialized(): 
        dist.destroy_process_group()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    local_rank, global_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    tb_writer = None
    if global_rank == 0:
        print(f"=== Training Started: {cfg.logger.run_name} (1M Data Optimized) ===")
        print(OmegaConf.to_yaml(cfg))
        os.makedirs(cfg.training.save_dir, exist_ok=True)
        
        if cfg.logger.use_tensorboard:
            tb_writer = SummaryWriter(log_dir=os.path.join("runs", cfg.logger.run_name))
        if cfg.logger.use_wandb:
            wandb.init(project=cfg.logger.project_name, name=cfg.logger.run_name, config=OmegaConf.to_container(cfg, resolve=True))

    history_len = cfg.model.get("history_len", 1)
    max_rollout = cfg.training.curriculum.max_rollout_steps
    noise_std = cfg.training.get("noise_std", 0.0) 
    
    seq_len_needed = max_rollout + history_len

    train_dataset = SapienSequenceDataset(data_path=cfg.dataset.train_path, sub_seq_len=seq_len_needed)
    val_dataset = SapienSequenceDataset(data_path=cfg.dataset.val_path, sub_seq_len=seq_len_needed)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, sampler=train_sampler, num_workers=cfg.training.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, sampler=val_sampler, num_workers=cfg.training.num_workers, pin_memory=True)

    pos_mean = train_dataset.pos_mean.to(device)
    pos_std = train_dataset.pos_std.to(device)
    vel_std = train_dataset.vel_std.to(device)
    
    model = SE3WorldModel(
        num_points=cfg.model.num_points, 
        latent_dim=cfg.model.latent_dim, 
        num_global_vectors=2, # 【修复1】：重力 (1) + 风力 (1)
        context_dim=1,        # 【修复1】：占位符标量维度
        history_len=history_len
    ).to(device)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)
    scaler = amp.GradScaler('cuda')
    
    criterion_pos = GeometricConsistencyLoss(
        lambda_rigid=cfg.training.loss_weights.rigid, 
        lambda_energy=cfg.training.loss_weights.energy
    ).to(device)

    best_val_loss = float("inf")
    global_step = 0
    
    for epoch in range(cfg.training.epochs):
        train_sampler.set_epoch(epoch)
        
        curr_config = cfg.training.curriculum
        current_rollout_steps = min(
            curr_config.max_rollout_steps, 
            curr_config.start_rollout_steps + epoch // curr_config.increase_every_epochs
        )
        
        tf_decay_steps = float(curr_config.teacher_forcing_decay_epochs)
        teacher_forcing_ratio = max(0.0, 1.0 - epoch / tf_decay_steps)
        
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (x_seq, v_seq, f_seq, explicit_seq, context_seq) in enumerate(train_loader):
            x_seq = x_seq.to(device, non_blocking=True)
            v_seq = v_seq.to(device, non_blocking=True)
            explicit_seq = explicit_seq.to(device, non_blocking=True)
            context_seq = context_seq.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            curr_x_hist = x_seq[:, 0:history_len] 
            curr_v_hist = v_seq[:, 0:history_len]
            
            batch_loss = 0.0
            
            for t in range(current_rollout_steps):
                target_idx = history_len + t
                target_v = v_seq[:, target_idx]
                target_x = x_seq[:, target_idx] 
                
                ctx_idx = target_idx - 1
                curr_expl = explicit_seq[:, ctx_idx]
                curr_ctx = context_seq[:, ctx_idx]
                
                # 【核心修复 2】：将风力转化为明确的 SE(3) 等变向量
                wind_expl = curr_ctx.unsqueeze(1)    # [B, 1, 3]
                combined_expl = torch.cat([curr_expl, wind_expl], dim=1) # [B, 2, 3]
                dummy_ctx = torch.ones(x_seq.shape[0], 1, device=device)

                if noise_std > 0:
                    input_x_hist = curr_x_hist.clone() + torch.randn_like(curr_x_hist) * noise_std
                    input_v_hist = curr_v_hist.clone() + torch.randn_like(curr_v_hist) * noise_std
                else:
                    input_x_hist = curr_x_hist
                    input_v_hist = curr_v_hist

                with amp.autocast('cuda'):
                    pred_v_norm, _, _, _ = model(
                        input_x_hist, input_v_hist, combined_expl, dummy_ctx, 
                        vel_std=vel_std, pos_mean=pos_mean, pos_std=pos_std
                    )
                
                pred_v_norm = pred_v_norm.float()
                
                last_x = curr_x_hist[:, -1] 
                pred_v_real = pred_v_norm * vel_std 
                next_x = last_x + pred_v_real / pos_std
                
                loss_total, _, _, _ = criterion_pos(next_x, target_x, pred_v_norm, target_v)
                step_loss = cfg.training.loss_weights.pos * loss_total
                batch_loss += step_loss
                
                if torch.rand(1).item() < teacher_forcing_ratio:
                    next_v_in, next_x_in = target_v, target_x
                else:
                    next_v_in, next_x_in = pred_v_norm, next_x

                curr_x_hist = torch.cat([curr_x_hist[:, 1:], next_x_in.unsqueeze(1)], dim=1)
                curr_v_hist = torch.cat([curr_v_hist[:, 1:], next_v_in.unsqueeze(1)], dim=1)

            final_loss = batch_loss / current_rollout_steps
            
            scaler.scale(final_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += final_loss.item()
            
            if global_rank == 0 and global_step % 20 == 0:
                if cfg.logger.use_wandb:
                    wandb.log({
                        "train/loss": final_loss.item(),
                        "train/rollout_steps": current_rollout_steps,
                        "train/tf_ratio": teacher_forcing_ratio,
                        "train/lr": optimizer.param_groups[0]['lr']
                    }, step=global_step)
            global_step += 1

        avg_train_loss = epoch_loss / len(train_loader)
        scheduler.step()

        model.eval()
        val_loss_total = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for x_seq, v_seq, f_seq, explicit_seq, context_seq in val_loader:
                x_seq = x_seq.to(device, non_blocking=True)
                v_seq = v_seq.to(device, non_blocking=True)
                explicit_seq = explicit_seq.to(device, non_blocking=True)
                context_seq = context_seq.to(device, non_blocking=True)

                curr_x_hist = x_seq[:, 0:history_len] 
                curr_v_hist = v_seq[:, 0:history_len]
                
                batch_loss = 0.0
                
                for t in range(current_rollout_steps):
                    target_idx = history_len + t
                    target_v = v_seq[:, target_idx]
                    target_x = x_seq[:, target_idx] 
                    
                    ctx_idx = target_idx - 1
                    curr_expl = explicit_seq[:, ctx_idx]
                    curr_ctx = context_seq[:, ctx_idx]
                    
                    # 【核心修复 3】：验证集同样对齐数据维度
                    wind_expl = curr_ctx.unsqueeze(1)
                    combined_expl = torch.cat([curr_expl, wind_expl], dim=1)
                    dummy_ctx = torch.ones(x_seq.shape[0], 1, device=device)
                    
                    with amp.autocast('cuda'):
                        pred_v_norm, _, _, _ = model(
                            curr_x_hist, curr_v_hist, combined_expl, dummy_ctx, 
                            vel_std=vel_std, pos_mean=pos_mean, pos_std=pos_std
                        )
                    pred_v_norm = pred_v_norm.float()
                    
                    last_x = curr_x_hist[:, -1] 
                    pred_v_real = pred_v_norm * vel_std 
                    next_x = last_x + pred_v_real / pos_std
                    
                    loss_total, _, _, _ = criterion_pos(next_x, target_x, pred_v_norm, target_v)
                    step_loss = (cfg.training.loss_weights.pos * loss_total)
                    batch_loss += step_loss
                    
                    next_v_in, next_x_in = pred_v_norm, next_x

                    curr_x_hist = torch.cat([curr_x_hist[:, 1:], next_x_in.unsqueeze(1)], dim=1)
                    curr_v_hist = torch.cat([curr_v_hist[:, 1:], next_v_in.unsqueeze(1)], dim=1)

                val_loss_total += (batch_loss / current_rollout_steps).item()
                val_steps += 1

        val_metrics = torch.tensor([val_loss_total, val_steps], device=device)
        if world_size > 1:
            dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
        
        avg_val_loss = val_metrics[0].item() / max(val_metrics[1].item(), 1)

        if global_rank == 0:
            print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            if cfg.logger.use_wandb:
                wandb.log({
                    "val/loss": avg_val_loss,
                    "epoch": epoch + 1
                }, step=global_step)
            
            latest_path = os.path.join(cfg.training.save_dir, "model_latest.pth")
            torch.save(model.module.state_dict(), latest_path)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(cfg.training.save_dir, "model_best.pth")
                torch.save(model.module.state_dict(), best_path)
                print(f"🌟 新的最优模型已保存! (Val Loss: {best_val_loss:.6f})")

    if global_rank == 0:
        if cfg.logger.use_wandb: wandb.finish()
    cleanup_distributed()

if __name__ == "__main__":
    main()

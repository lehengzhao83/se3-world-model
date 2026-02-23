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

# 将 src 目录添加到 Python 路径，以便能导入自定义模块
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from se3_world_model.dataset import SapienSequenceDataset
from se3_world_model.model import SE3WorldModel
from se3_world_model.loss import GeometricConsistencyLoss

def setup_distributed():
    """
    初始化分布式训练环境 (DDP Setup)。
    
    检测环境变量 'RANK' 和 'WORLD_SIZE'，这些通常由 torchrun 命令自动设置。
    如果是单卡运行，这些变量不存在，则默认回退到单进程模式。
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])            # 当前进程的全局序号 (Global Rank)
        world_size = int(os.environ["WORLD_SIZE"]) # 总进程数 (Total GPU count)
        local_rank = int(os.environ["LOCAL_RANK"]) # 当前节点上的序号 (Local Rank, 单机多卡时通常等于 GPU ID)
        
        # 初始化进程组，使用 NCCL 后端 (NVIDIA GPU 通信的最佳实践)
        dist.init_process_group(backend="nccl", init_method="env://")
        # 设置当前进程使用的 GPU 设备
        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size
    
    # 如果不是分布式环境，默认返回 0, 0, 1 (单卡模式)
    return 0, 0, 1

def cleanup_distributed():
    """
    清理分布式进程组资源，防止程序退出时挂起。
    """
    if dist.is_initialized(): dist.destroy_process_group()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    主训练函数。
    使用 Hydra 管理配置，配置内容会自动注入到 cfg 参数中。
    """
    # 1. 设置分布式环境
    local_rank, global_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # 2. 初始化日志记录器 (仅在主进程 global_rank == 0 执行)
    tb_writer = None
    if global_rank == 0:
        print(f"=== Training Started: {cfg.logger.run_name} ===")
        print(OmegaConf.to_yaml(cfg)) # 打印当前配置
        os.makedirs(cfg.training.save_dir, exist_ok=True)
        
        # 初始化 TensorBoard
        if cfg.logger.use_tensorboard:
            tb_writer = SummaryWriter(log_dir=os.path.join("runs", cfg.logger.run_name))
        # 初始化 Weights & Biases (WandB)
        if cfg.logger.use_wandb:
            wandb.init(project=cfg.logger.project_name, name=cfg.logger.run_name, config=OmegaConf.to_container(cfg, resolve=True))

    # 3. 读取关键超参数
    history_len = cfg.model.get("history_len", 1)
    max_rollout = cfg.training.curriculum.max_rollout_steps
    noise_std = cfg.training.get("noise_std", 0.0) # 训练时的噪声注入强度，增加鲁棒性

    # 4. 数据集初始化
    # sub_seq_len = 最大预测步数 + 历史长度，保证切分出的序列足够长
    dataset = SapienSequenceDataset(
        data_path=cfg.dataset.train_path, 
        sub_seq_len=max_rollout + history_len
    )
    
    # 使用 DistributedSampler 确保每个 GPU 处理数据集的不同部分，互不重叠
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.training.batch_size, 
        sampler=sampler, 
        num_workers=cfg.training.num_workers, 
        pin_memory=True # 开启锁页内存，加速 CPU 到 GPU 的数据传输
    )

    # 5. 加载数据标准化统计量 (均值/方差) 并移至 GPU
    # 这些统计量用于在 Loss 计算时将预测的归一化值还原为真实物理值
    pos_std = dataset.pos_std.to(device)
    vel_std = dataset.vel_std.to(device)
    vel_mean = dataset.vel_mean.to(device)
    
    # 6. 模型初始化
    model = SE3WorldModel(
        num_points=cfg.model.num_points, 
        latent_dim=cfg.model.latent_dim, 
        num_global_vectors=cfg.model.num_global_vectors, 
        context_dim=cfg.model.context_dim,
        history_len=history_len
    ).to(device)
    
    # 使用 DDP 包装模型，实现梯度的自动同步
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # 优化器设置
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    
    # 损失函数设置
    criterion_vel = nn.MSELoss() # 速度损失使用标准 MSE
    # 位置损失使用自定义的几何一致性损失 (包含刚性约束、能量约束等)
    criterion_pos = GeometricConsistencyLoss(
        lambda_rigid=cfg.training.loss_weights.rigid, 
        lambda_energy=cfg.training.loss_weights.energy
    ).to(device)

    model.train()

    global_step = 0
    
    # === 开始训练循环 ===
    for epoch in range(cfg.training.epochs):
        # DDP 必须：每个 epoch 设置 sampler 的 epoch，以保证 shuffle 的随机种子变化
        sampler.set_epoch(epoch)
        
        # === 课程学习 (Curriculum Learning) ===
        # 随着训练进行，逐步增加 rollout (向后预测) 的步数。
        # 刚开始只预测 1 步，模型稳定后预测更多步，防止累积误差过早导致训练崩溃。
        curr_config = cfg.training.curriculum
        current_rollout_steps = min(
            curr_config.max_rollout_steps, 
            curr_config.start_rollout_steps + epoch // curr_config.increase_every_epochs
        )
        
        # === 教师强制衰减 (Teacher Forcing Decay) ===
        # teacher_forcing_ratio: 有多大的概率使用"真实历史"作为下一步的输入。
        # 训练初期接近 1.0 (全用真值)，后期逐渐降为 0.0 (全用模型自己的预测值)，
        # 以弥补训练和推理时的分布偏移 (Exposure Bias)。
        tf_decay_steps = float(curr_config.teacher_forcing_decay_epochs)
        teacher_forcing_ratio = max(0.0, 1.0 - epoch / tf_decay_steps)
        
        epoch_loss = 0.0
        
        for batch_idx, (x_seq, v_seq, explicit_seq, context_seq) in enumerate(dataloader):
            # 将数据移动到 GPU，non_blocking=True 实现异步传输
            x_seq = x_seq.to(device, non_blocking=True)
            v_seq = v_seq.to(device, non_blocking=True)
            explicit_seq = explicit_seq.to(device, non_blocking=True)
            context_seq = context_seq.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # 初始化历史窗口
            # curr_x_hist 维度: [B, history_len, N, 3]
            curr_x_hist = x_seq[:, 0:history_len] 
            curr_v_hist = v_seq[:, 0:history_len]
            
            batch_loss = 0.0
            total_rigid_loss = 0.0
            
            # === 自回归 Rollout 循环 ===
            # 从 t=0 预测到 t=current_rollout_steps
            for t in range(current_rollout_steps):
                # 确定当前预测目标的时间步索引
                target_idx = history_len + t
                target_v = v_seq[:, target_idx] # 目标速度 Ground Truth
                target_x = x_seq[:, target_idx] # 目标位置 Ground Truth
                
                # 获取当前步对应的上下文信息 (explicit/context 通常与当前时刻对齐)
                ctx_idx = target_idx - 1
                curr_expl = explicit_seq[:, ctx_idx]
                curr_ctx = context_seq[:, ctx_idx]
                
                # === 噪声注入 (Noise Injection) ===
                # 在输入历史中加入微小的高斯噪声，防止模型过拟合，提高抗扰动能力
                if model.training and noise_std > 0:
                    input_x_hist = curr_x_hist.clone() + torch.randn_like(curr_x_hist) * noise_std
                    input_v_hist = curr_v_hist.clone() + torch.randn_like(curr_v_hist) * noise_std
                else:
                    input_x_hist = curr_x_hist
                    input_v_hist = curr_v_hist

                # === 模型前向传播 ===
                # pred_v_norm: 归一化后的预测速度
                pred_v_norm, _ = model(input_x_hist, input_v_hist, curr_expl, curr_ctx)
                
                # === 物理状态更新 (欧拉积分) ===
                # 为了计算位置 Loss，需要将归一化速度还原并积分到位置上
                last_x = curr_x_hist[:, -1] # 上一步的位置
                # 反归一化：pred_v_real = pred_v_norm * std + mean
                pred_v_real = pred_v_norm * vel_std + vel_mean
                # 积分：下一位置 = 上一位置 + 速度 (注意位置也需要除以 pos_std 变回归一化空间)
                next_x = last_x + pred_v_real / pos_std
                
                # === 损失计算 ===
                # 1. 速度直接误差 (MSE)
                loss_v = criterion_vel(pred_v_norm, target_v)
                # 2. 位置几何一致性误差 (包含位置MSE, 刚体性, 能量守恒等)
                loss_x_total, _, loss_rigid, loss_energy = criterion_pos(next_x, target_x, pred_v_norm, target_v)
                
                # 组合单步 Loss
                step_loss = loss_v + (cfg.training.loss_weights.pos * loss_x_total)
                batch_loss += step_loss
                total_rigid_loss += loss_rigid
                
                # === 教师强制决策 (Teacher Forcing) ===
                # 决定下一步的输入是使用"模型刚才预测的" (Autoregressive) 
                # 还是使用"数据集里的真实值" (Ground Truth)
                use_ground_truth = (torch.rand(1).item() < teacher_forcing_ratio)
                
                if use_ground_truth:
                    next_v_in = target_v
                    next_x_in = target_x
                else:
                    # 如果使用预测值，必须 detach 梯度，防止 BPTT (Backpropagation Through Time) 
                    # 梯度链过长导致显存爆炸或梯度不稳定 (此处采用截断式 BPTT 的变体)
                    next_v_in = pred_v_norm.detach() 
                    next_x_in = next_x.detach()

                # === 更新滑动窗口 ===
                # 移除最旧的一帧，加入最新的一帧 (Real or Pred)
                # curr_x_hist: [B, H, N, 3] -> 移除 index 0, 拼接 new at end
                curr_x_hist = torch.cat([curr_x_hist[:, 1:], next_x_in.unsqueeze(1)], dim=1)
                curr_v_hist = torch.cat([curr_v_hist[:, 1:], next_v_in.unsqueeze(1)], dim=1)

            # === 反向传播 ===
            # 对整个 Rollout 序列的平均 Loss 进行优化
            final_loss = batch_loss / current_rollout_steps
            final_loss.backward()
            
            # 梯度裁剪：防止梯度爆炸，这对训练深层网络和序列模型至关重要
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.grad_clip)
            optimizer.step()
            
            epoch_loss += final_loss.item()
            
            # === 日志记录 ===
            if global_rank == 0 and global_step % 10 == 0:
                if cfg.logger.use_wandb:
                    wandb.log({
                        "train_loss": final_loss.item(),
                        "rollout_steps": current_rollout_steps,
                        "tf_ratio": teacher_forcing_ratio
                    }, step=global_step)
            global_step += 1

        # === Epoch 结束处理 ===
        avg_epoch_loss = epoch_loss / len(dataloader)
        if global_rank == 0:
            print(f"Epoch {epoch+1} done. Avg Loss: {avg_epoch_loss:.6f}")
            # 保存模型权重
            save_path = os.path.join(cfg.training.save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.module.state_dict(), save_path)

    # 训练结束，关闭 WandB 和分布式组
    if global_rank == 0:
        if cfg.logger.use_wandb: wandb.finish()
    cleanup_distributed()

if __name__ == "__main__":
    main()

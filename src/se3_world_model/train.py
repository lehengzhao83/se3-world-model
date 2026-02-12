import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from se3_world_model.dataset import SapienDataset
from se3_world_model.model import SE3WorldModel


def setup_distributed():
    """初始化分布式训练环境"""
    # torchrun 会自动设置这些环境变量
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        # 初始化进程组 (使用 NCCL 后端，针对 NVIDIA GPU 优化)
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        
        return local_rank, rank, world_size
    else:
        print("未检测到分布式环境，将使用单卡模式。")
        return 0, 0, 1


def cleanup_distributed():
    """销毁分布式进程组"""
    if dist.is_initialized():
        dist.destroy_process_group()


def train(
    epochs: int = 10,
    batch_size: int = 16,  # 注意：这是单卡 batch size
    lr: float = 1e-3,
    save_dir: str = "checkpoints"
) -> None:
    
    # 1. 分布式设置
    local_rank, global_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0:
        print(f"启动分布式训练：检测到 {world_size} 张显卡")
        print(f"每张卡的 Batch Size: {batch_size} (总 Batch Size: {batch_size * world_size})")

    # 2. 数据准备
    # 为了防止 8 个进程同时生成数据导致 CPU/内存爆炸，我们可以让主进程先生成，
    # 或者如果数据生成很快，就各自生成。SAPIEN 比较轻量，我们各自生成即可。
    if global_rank == 0:
        print("初始化 SAPIEN 仿真环境...")
    
    # 注意：这里我们适当减少单卡的样本量，因为总样本量 = num_samples * world_size (近似)
    # 或者保持不变，这样总数据量就是 8 倍，训练更充分
    dataset = SapienDataset(data_path="data/sapien_train.pt")
    
    # 关键：使用 DistributedSampler 确保每张卡分到不同的数据
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=4,        # 4090 性能强劲，可以多开几个 worker 加速数据加载
        pin_memory=True       # 加速 CPU -> GPU 传输
    )
    
    # 3. 模型初始化
    model = SE3WorldModel(
        num_points=64,
        latent_dim=64,
        num_global_vectors=1,
        context_dim=3
    ).to(device)

    # 关键：将模型封装为 DDP 模型
    # find_unused_parameters=False 通常能提升性能，除非你的模型有些层在 forward 里没用到
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 4. 训练循环
    if global_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        print(f"开始训练，共 {epochs} 个 Epoch...")

    model.train()

    for epoch in range(epochs):
        # 关键：每个 epoch 开始前设置 sampler 的 epoch，保证每个 epoch 的 shuffle 都不一样
        sampler.set_epoch(epoch)
        
        total_loss = 0.0
        # dataloader 已经是切分后的数据了
        for batch_idx, (x_t, explicit, context, x_next) in enumerate(dataloader):
            x_t = x_t.to(device, non_blocking=True)
            explicit = explicit.to(device, non_blocking=True)
            context = context.to(device, non_blocking=True)
            x_next = x_next.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred_next, _ = model(x_t, explicit, context)

            loss = criterion(pred_next, x_next)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 只在主进程打印日志，避免刷屏
            if global_rank == 0 and batch_idx % 10 == 0:
                print(f"[GPU 0] Epoch [{epoch+1}/{epochs}] Step [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.6f}")

        # 计算平均 Loss (这里只打印 Rank 0 的 Loss 作为参考)
        avg_loss = total_loss / len(dataloader)
        
        if global_rank == 0:
            print(f"=== Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.6f} ===")
            # 只在主进程保存模型
            # 注意：保存 model.module 而不是 model 本身，因为 model 被 DDP 包裹了
            torch.save(model.module.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))

    if global_rank == 0:
        print("训练完成。")
    
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50) # 8卡跑得快，可以多跑几轮
    parser.add_argument("--batch_size", type=int, default=64) # 4090 显存大，单卡设为 64，总 Batch Size 就是 512
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

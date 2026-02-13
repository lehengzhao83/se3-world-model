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

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from se3_world_model.dataset import SapienDataset  # noqa: E402
from se3_world_model.model import SE3WorldModel  # noqa: E402


def setup_distributed() -> tuple[int, int, int]:
    """初始化分布式训练环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

        return local_rank, rank, world_size
    else:
        print("未检测到分布式环境，将使用单卡模式。")
        return 0, 0, 1


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def train(
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    save_dir: str = "checkpoints"
) -> None:

    local_rank, global_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0:
        print(f"启动分布式训练：检测到 {world_size} 张显卡")

    if global_rank == 0:
        print("初始化 SAPIEN 仿真环境...")

    dataset = SapienDataset(data_path="data/sapien_train.pt")
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    model = SE3WorldModel(
        num_points=64,
        latent_dim=64,
        num_global_vectors=1,
        context_dim=3
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if global_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        print(f"开始训练，共 {epochs} 个 Epoch...")

    model.train()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        total_loss = 0.0
        
        # 解包新增的 v_t
        for batch_idx, (x_t, v_t, explicit, context, x_next) in enumerate(dataloader):
            x_t = x_t.to(device, non_blocking=True)
            v_t = v_t.to(device, non_blocking=True) # New
            explicit = explicit.to(device, non_blocking=True)
            context = context.to(device, non_blocking=True)
            x_next = x_next.to(device, non_blocking=True)

            optimizer.zero_grad()
            # 传入速度 v_t
            pred_next, _ = model(x_t, v_t, explicit, context)

            loss = criterion(pred_next, x_next)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if global_rank == 0 and batch_idx % 10 == 0:
                print(f"[GPU 0] Epoch [{epoch + 1}/{epochs}] Step [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)

        if global_rank == 0:
            print(f"=== Epoch [{epoch + 1}/{epochs}] Average Loss: {avg_loss:.6f} ===")
            torch.save(model.module.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth"))

    if global_rank == 0:
        print("训练完成。")

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

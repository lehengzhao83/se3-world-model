# ... (Imports same as before)
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

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from se3_world_model.dataset import SapienDataset
from se3_world_model.model import SE3WorldModel

# ... (setup_distributed, cleanup_distributed same as before)
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

def train(epochs: int = 50, batch_size: int = 64, lr: float = 1e-3, save_dir: str = "checkpoints"):
    local_rank, global_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0: print("Initializing...")

    # Load Dataset (Computes Vel Stats automatically)
    dataset = SapienDataset(data_path="data/sapien_train.pt")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    model = SE3WorldModel(num_points=64, latent_dim=64, num_global_vectors=1, context_dim=3).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if global_rank == 0: os.makedirs(save_dir, exist_ok=True)

    model.train()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        total_loss = 0.0
        
        # Note: Unpacking target_v_norm instead of x_next
        for batch_idx, (x_norm, v_norm, explicit, context, target_v_norm) in enumerate(dataloader):
            x_norm = x_norm.to(device, non_blocking=True)
            v_norm = v_norm.to(device, non_blocking=True)
            explicit = explicit.to(device, non_blocking=True)
            context = context.to(device, non_blocking=True)
            target_v_norm = target_v_norm.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # Predict Normalized Velocity
            pred_v_norm, _ = model(x_norm, v_norm, explicit, context)

            # Loss on Velocity (Scales are now O(1))
            loss = criterion(pred_v_norm, target_v_norm)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if global_rank == 0 and batch_idx % 10 == 0:
                print(f"[GPU 0] Epoch {epoch+1} Step {batch_idx} Loss: {loss.item():.6f}")

        if global_rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"=== Epoch {epoch+1} Avg Loss: {avg_loss:.6f} ===")
            torch.save(model.module.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))

    cleanup_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

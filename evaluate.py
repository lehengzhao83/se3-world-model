import os
import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
# === 修改点: ===
from se3_world_model.dataset import SapienSequenceDataset
from se3_world_model.model import SE3WorldModel

def evaluate(checkpoint_path, data_path="data/sapien_val_seq.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading validation data from {data_path}...")
    
    # 1. 加载验证集 (Sequence)
    # 评估时我们可以只取 H=1 或者 H=5，这里取 H=1 做单步验证
    dataset = SapienSequenceDataset(data_path=data_path, sub_seq_len=2)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    pos_mean = dataset.pos_mean.to(device)
    pos_std = dataset.pos_std.to(device)
    vel_mean = dataset.vel_mean.to(device)
    vel_std = dataset.vel_std.to(device)
    
    print(f"Loading model from {checkpoint_path}...")
    model = SE3WorldModel(
        num_points=64,
        latent_dim=64,
        num_global_vectors=1,
        context_dim=3
    ).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    
    total_mse = 0.0
    vis_input, vis_target, vis_pred = None, None, None
    
    with torch.no_grad():
        # 数据集返回: x_seq, v_seq, explicit, context
        for i, (x_seq, v_seq, explicit, context) in enumerate(dataloader):
            # 取 t=0 作为输入，t=1 作为目标
            x_t = x_seq[:, 0].to(device)
            v_t = v_seq[:, 0].to(device)
            target_v_next = v_seq[:, 1].to(device) # 目标是速度
            
            expl = explicit[:, 0].to(device)
            ctx = context[:, 0].to(device)
            
            # 预测归一化速度
            pred_v_next, _ = model(x_t, v_t, expl, ctx)
            
            mse = torch.nn.functional.mse_loss(pred_v_next, target_v_next)
            total_mse += mse.item()
            
            if i == 0:
                # 可视化位置 (积分一步)
                # ratio = vel_std / pos_std
                ratio = vel_std / pos_std
                
                # Input Pos
                vis_input = (x_t * pos_std + pos_mean).cpu()
                
                # Target Pos (x_t + v_real)
                # 或者直接用 dataset 里的 x_{t+1}
                target_x = x_seq[:, 1].to(device)
                vis_target = (target_x * pos_std + pos_mean).cpu()
                
                # Pred Pos (Integrated)
                pred_x_norm = x_t + pred_v_next * ratio
                vis_pred = (pred_x_norm * pos_std + pos_mean).cpu()

    avg_mse = total_mse / len(dataloader)
    print(f"Validation Velocity MSE: {avg_mse:.6f}")
    
    if vis_input is not None:
        visualize(vis_input[0], vis_target[0], vis_pred[0])

def visualize(input_p, target_p, pred_p):
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(input_p[:,0], input_p[:,1], input_p[:,2], c='blue', alpha=0.3, label='Input')
    ax1.scatter(target_p[:,0], target_p[:,1], target_p[:,2], c='green', label='Target')
    ax1.scatter(pred_p[:,0], pred_p[:,1], pred_p[:,2], c='red', marker='x', label='Pred')
    ax1.legend()
    
    save_path = "assets/eval_result.png"
    os.makedirs("assets", exist_ok=True)
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    evaluate("checkpoints/model_epoch_50.pth")

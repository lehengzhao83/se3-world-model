import os
import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 引入你的代码
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from se3_world_model.dataset import SapienDataset
from se3_world_model.model import SE3WorldModel

def evaluate(checkpoint_path, data_path="data/sapien_val.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading validation data from {data_path}...")
    
    # 1. 加载验证集 (Dataset 已经修改为返回 v_t)
    dataset = SapienDataset(data_path=data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 2. 加载模型
    print(f"Loading model from {checkpoint_path}...")
    model = SE3WorldModel(
        num_points=64,
        latent_dim=64,
        num_global_vectors=1,
        context_dim=3
    ).to(device)
    
    # 加载权重
    state_dict = torch.load(checkpoint_path, map_location=device)
    # 去除 DDP 的 'module.' 前缀
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # 3. 推理与计算误差
    total_mse = 0.0
    
    # 用于可视化的数据
    vis_input = None
    vis_target = None
    vis_pred = None
    
    with torch.no_grad():
        # 修改点：解包增加 v_t
        for i, (x_t, v_t, explicit, context, x_next) in enumerate(dataloader):
            x_t = x_t.to(device)
            v_t = v_t.to(device) # 新增：移动速度到 GPU
            explicit = explicit.to(device)
            context = context.to(device)
            x_next = x_next.to(device)
            
            # 修改点：传入速度 v_t
            pred_next, _ = model(x_t, v_t, explicit, context)
            
            # 计算误差
            mse = torch.nn.functional.mse_loss(pred_next, x_next)
            total_mse += mse.item()
            
            # 收集第一个 Batch 用于可视化
            if i == 0:
                vis_pred = pred_next.cpu()
                vis_target = x_next.cpu()
                vis_input = x_t.cpu()

    avg_mse = total_mse / len(dataloader)
    print(f"Validation MSE: {avg_mse:.6f}")
    
    # 4. 可视化 (画出第一个样本的对比)
    if vis_input is not None:
        visualize(vis_input[0], vis_target[0], vis_pred[0])

def visualize(input_p, target_p, pred_p):
    """保存一张对比图：输入(蓝) -> 真实(绿) vs 预测(红)"""
    fig = plt.figure(figsize=(12, 5))
    
    # 视角 1
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(input_p[:,0], input_p[:,1], input_p[:,2], c='blue', alpha=0.3, label='Input (t)')
    ax1.scatter(target_p[:,0], target_p[:,1], target_p[:,2], c='green', marker='o', label='Target (t+1)')
    ax1.scatter(pred_p[:,0], pred_p[:,1], pred_p[:,2], c='red', marker='x', label='Pred (t+1)')
    ax1.set_title("3D View")
    ax1.legend()
    
    # 视角 2 (投影)
    ax2 = fig.add_subplot(122)
    ax2.scatter(input_p[:,0], input_p[:,1], c='blue', alpha=0.3, label='Input')
    ax2.scatter(target_p[:,0], target_p[:,1], c='green', alpha=0.6, label='Target')
    ax2.scatter(pred_p[:,0], pred_p[:,1], c='red', marker='x', label='Pred')
    ax2.set_title("2D Projection (XY)")
    ax2.legend()
    
    save_path = "assets/eval_result.png"
    os.makedirs("assets", exist_ok=True)
    plt.savefig(save_path)
    print(f"\nVisualization saved to: {save_path}")

if __name__ == "__main__":
    # 使用最后一个 Epoch 的模型
    evaluate("checkpoints/model_epoch_50.pth")

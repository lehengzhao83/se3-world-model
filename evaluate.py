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
    
    # 1. 加载验证集
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
    
    # 加载权重 (注意：因为训练时用了 DDP，权重键值可能有 'module.' 前缀，需要处理)
    state_dict = torch.load(checkpoint_path, map_location=device)
    # 去除 'module.' 前缀
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # 3. 推理与计算误差
    total_mse = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, (x_t, explicit, context, x_next) in enumerate(dataloader):
            x_t = x_t.to(device)
            explicit = explicit.to(device)
            context = context.to(device)
            x_next = x_next.to(device)
            
            # 预测
            pred_next, _ = model(x_t, explicit, context)
            
            # 计算误差
            mse = torch.nn.functional.mse_loss(pred_next, x_next)
            total_mse += mse.item()
            
            # 收集第一个 Batch 用于可视化
            if i == 0:
                all_preds = pred_next.cpu()
                all_targets = x_next.cpu()
                input_points = x_t.cpu()

    avg_mse = total_mse / len(dataloader)
    print(f"Validation MSE: {avg_mse:.6f}")
    
    # 4. 可视化 (画出第一个样本的对比)
    visualize(input_points[0], all_targets[0], all_preds[0])

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
    
    save_path = "eval_result.png"
    plt.savefig(save_path)
    print(f"\nVisualization saved to: {save_path}")
    print("你可以通过 scp 将这张图片下载到本地查看。")

if __name__ == "__main__":
    # 使用最后一个 Epoch 的模型
    evaluate("checkpoints/model_epoch_50.pth")

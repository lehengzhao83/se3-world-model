import os
import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from se3_world_model.dataset import SapienDataset
from se3_world_model.model import SE3WorldModel

def evaluate(checkpoint_path, data_path="data/sapien_val.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading validation data from {data_path}...")
    
    # 1. 加载验证集
    dataset = SapienDataset(data_path=data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 获取反归一化参数
    pos_mean = dataset.pos_mean.to(device)
    pos_std = dataset.pos_std.to(device)
    
    # 2. 加载模型
    print(f"Loading model from {checkpoint_path}...")
    model = SE3WorldModel(
        num_points=64, 
        latent_dim=64, 
        num_global_vectors=1, 
        context_dim=3
    ).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    
    vis_input, vis_target, vis_pred = None, None, None
    
    with torch.no_grad():
        for i, (x_t, v_t, explicit, context, x_next) in enumerate(dataloader):
            x_t, v_t, explicit, context, x_next = x_t.to(device), v_t.to(device), explicit.to(device), context.to(device), x_next.to(device)
            
            # 预测 (输出的是 Normalized 坐标)
            pred_next, _ = model(x_t, v_t, explicit, context)
            
            if i == 0:
                # 反归一化后保存用于画图
                vis_input = (x_t * pos_std + pos_mean).cpu()
                vis_target = (x_next * pos_std + pos_mean).cpu()
                vis_pred = (pred_next * pos_std + pos_mean).cpu()
                break

    # 4. 可视化
    visualize(vis_input[0], vis_target[0], vis_pred[0])

def visualize(input_p, target_p, pred_p):
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(input_p[:,0], input_p[:,1], input_p[:,2], c='blue', alpha=0.3, label='Input')
    ax1.scatter(target_p[:,0], target_p[:,1], target_p[:,2], c='green', label='Target')
    ax1.scatter(pred_p[:,0], pred_p[:,1], pred_p[:,2], c='red', marker='x', label='Pred')
    ax1.set_title("3D View (Real Coords)")
    ax1.legend()
    
    save_path = "assets/eval_result.png"
    os.makedirs("assets", exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    evaluate("checkpoints/model_epoch_50.pth")

import os
import sys
import numpy as np
import sapien.core as sapien
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 强制 Matplotlib 使用非交互式后端
plt.switch_backend('Agg')

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
# === 修改点 1: 引用新的 Dataset 类 ===
from se3_world_model.dataset import SapienSequenceDataset
from se3_world_model.model import SE3WorldModel

def sample_capsule_points(r: float, l: float, n: int) -> np.ndarray:
    points = []
    while len(points) < n:
        pt = np.random.uniform(low=[-l - r, -r, -r], high=[l + r, r, r])
        px = pt[0]
        if px < -l:
            dist = np.linalg.norm(pt - np.array([-l, 0, 0]))
        elif px > l:
            dist = np.linalg.norm(pt - np.array([l, 0, 0]))
        else:
            dist = np.linalg.norm(pt[1:])
        if dist <= r:
            points.append(pt)
    return np.array(points, dtype=np.float32)

def make_rollout_video(checkpoint_path: str, save_path: str = "assets/simulation.gif"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # === 修改点 2: 加载统计量 (必须与训练时一致) ===
    print("Loading dataset stats...")
    # 注意：这里加载的是序列数据文件
    train_dataset = SapienSequenceDataset("data/sapien_train_seq.pt")
    
    pos_mean = train_dataset.pos_mean.to(device)
    pos_std = train_dataset.pos_std.to(device)
    vel_mean = train_dataset.vel_mean.to(device)
    vel_std = train_dataset.vel_std.to(device)
    
    # 计算积分比例因子: ratio = Vel_Std / Pos_Std
    # 因为 x_norm 和 v_norm 的缩放不同，dx_norm = v_norm * ratio
    scale_ratio = (vel_std / pos_std).view(1, 1, 3)

    print(f"Stats loaded. Scale Ratio: {scale_ratio[0,0,0].item():.4f}")

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

    print("Initializing SAPIEN simulation...")
    engine = sapien.Engine()
    engine.set_log_level("error")
    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    gravity_np = np.array([0.0, -2.0, -9.0], dtype=np.float32)
    wind_np = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    total_force = gravity_np + wind_np

    builder = scene.create_actor_builder()
    builder.add_capsule_collision(radius=0.1, half_length=0.2)
    builder.set_mass_and_inertia(1.0, sapien.Pose(), np.array([0.1, 0.1, 0.1], dtype=np.float32))
    actor = builder.build(name="dynamic_object")
    
    # 初始状态
    actor.set_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
    actor.set_velocity(np.array([0.0, 1.0, 2.0], dtype=np.float32)) 

    canonical_points = sample_capsule_points(0.1, 0.2, 64)

    # === Warmup: 获取初始 t=0 的状态 ===
    # T-1
    mat_prev = actor.get_pose().to_transformation_matrix()
    pts_prev = (mat_prev[:3, :3] @ canonical_points.T).T + mat_prev[:3, 3]

    for _ in range(5):
        actor.add_force_at_point(total_force, actor.get_pose().p)
        scene.step()

    # T (Current)
    mat_curr = actor.get_pose().to_transformation_matrix()
    pts_curr = (mat_curr[:3, :3] @ canonical_points.T).T + mat_curr[:3, 3]
    
    # 计算真实速度
    velocity_raw = pts_curr - pts_prev

    # === 归一化输入 ===
    pts_curr_tensor = torch.tensor(pts_curr).float().to(device).unsqueeze(0) # [1, N, 3]
    velocity_tensor = torch.tensor(velocity_raw).float().to(device).unsqueeze(0)
    
    # Apply Normalization
    curr_x_norm = (pts_curr_tensor - pos_mean) / pos_std
    curr_v_norm = (velocity_tensor - vel_mean) / vel_std
    
    explicit_input = torch.tensor(gravity_np).float().to(device).view(1, 1, 3)
    context_input = torch.tensor(wind_np).float().to(device).view(1, 3)

    frames = 60
    gt_trajectory = []
    pred_trajectory = []

    print("Running rollout...")
    for _ in range(frames):
        # --- A. GT (Real World) ---
        gt_trajectory.append(pts_curr) 

        for _ in range(5):
            actor.add_force_at_point(total_force, actor.get_pose().p)
            scene.step()
        
        mat_new = actor.get_pose().to_transformation_matrix()
        pts_new = (mat_new[:3, :3] @ canonical_points.T).T + mat_new[:3, 3]
        pts_curr = pts_new

        # --- B. Pred (Normalized Space Integration) ---
        with torch.no_grad():
            # 1. 预测下一帧的归一化速度
            pred_v_norm, _ = model(curr_x_norm, curr_v_norm, explicit_input, context_input)
            
            # 2. 积分更新状态 (与 train.py 逻辑完全一致)
            # x_{t+1} = x_t + v_{t+1} * ratio
            curr_x_norm = curr_x_norm + pred_v_norm * scale_ratio
            
            # v_{t+1} = pred_v_norm
            curr_v_norm = pred_v_norm
            
            # 3. 反归一化用于画图 (Denormalize)
            # x_real = x_norm * std + mean
            pred_real = curr_x_norm * pos_std + pos_mean
            pred_trajectory.append(pred_real.squeeze(0).cpu().numpy())

    # Rendering
    print(f"Rendering video to {save_path}...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    all_pts = np.concatenate(gt_trajectory + pred_trajectory)
    min_b, max_b = all_pts.min(0) - 1.0, all_pts.max(0) + 1.0

    def update(frame):
        ax.clear()
        ax.set_xlim(min_b[0], max_b[0])
        ax.set_ylim(min_b[1], max_b[1])
        ax.set_zlim(min_b[2], max_b[2])
        ax.set_title(f"Rollout Frame {frame}/{frames}\nGreen: GT | Red: Pred (Seq-Rollout)")
        
        gt = gt_trajectory[frame]
        ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='green', alpha=0.4, label='GT')
        
        pred = pred_trajectory[frame]
        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='red', marker='x', alpha=0.8, label='Pred')
        
        ax.legend()
        return ax,

    ani = FuncAnimation(fig, update, frames=frames, interval=100)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ani.save(save_path, writer='pillow', fps=10)
    print("Done!")

if __name__ == "__main__":
    make_rollout_video("checkpoints/model_epoch_50.pth")

import os
import sys
import numpy as np
import sapien.core as sapien
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 强制 Matplotlib 使用非交互式后端，防止无显示器报错
plt.switch_backend('Agg')

# 引入项目代码
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
try:
    from se3_world_model.model import SE3WorldModel
except ImportError:
    # 兼容性处理：如果目录结构不同，请确保能引用到模型
    print("Warning: Could not import SE3WorldModel. Ensure src/ is in PYTHONPATH.")

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
    print(f"Loading model from {checkpoint_path}...")

    # 1. 加载模型
    model = SE3WorldModel(
        num_points=64,
        latent_dim=64,
        num_global_vectors=1,
        context_dim=3
    ).to(device)
    
    # 加载权重
    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # 2. 初始化 SAPIEN 仿真环境
    print("Initializing SAPIEN simulation...")
    engine = sapien.Engine()
    engine.set_log_level("error")
    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    # 定义力 (SAPIEN 2.2.2 不支持 set_gravity，我们之后手动加)
    gravity_np = np.array([0.0, -2.0, -9.0], dtype=np.float32)
    wind_np = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    # 合力 (F = m*a, m=1.0)
    total_force = gravity_np + wind_np

    # 创建物体
    builder = scene.create_actor_builder()
    builder.add_capsule_collision(radius=0.1, half_length=0.2)
    # 匿名参数适配：质量, Pose, 惯性张量
    builder.set_mass_and_inertia(1.0, sapien.Pose(), np.array([0.1, 0.1, 0.1], dtype=np.float32))
    actor = builder.build(name="dynamic_object")
    
    # 初始姿态
    actor.set_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
    actor.set_velocity(np.array([0.0, 1.0, 1.0], dtype=np.float32)) 

    # 预计算标准点云
    canonical_points = sample_capsule_points(0.1, 0.2, 64)

    # 3. 开始推演 (Rollout)
    frames = 60 
    gt_trajectory = [] 
    pred_trajectory = [] 

    # 初始点云输入
    pose = actor.get_pose()
    mat = pose.to_transformation_matrix()
    cp_np = (mat[:3, :3] @ canonical_points.T).T + mat[:3, 3]
    current_pred_points = torch.tensor(cp_np).float().to(device).unsqueeze(0)

    # 准备输入向量
    explicit_input = torch.tensor(gravity_np).float().to(device).view(1, 1, 3)
    context_input = torch.tensor(wind_np).float().to(device).view(1, 3)

    print("Running rollout (Auto-regressive prediction)...")
    for _ in range(frames):
        # --- A. 获取 Ground Truth (物理模拟 5 步) ---
        for _ in range(5):
            # 获取当前位置作为施力点
            curr_p = actor.get_pose().p
            actor.add_force_at_point(total_force, curr_p)
            scene.step()
        
        pose = actor.get_pose()
        mat = pose.to_transformation_matrix()
        gt_points = (mat[:3, :3] @ canonical_points.T).T + mat[:3, 3]
        gt_trajectory.append(gt_points)

        # --- B. 模型预测 (自回归) ---
        with torch.no_grad():
            next_pred_points, _ = model(current_pred_points, explicit_input, context_input)
            pred_np = next_pred_points.squeeze(0).cpu().numpy()
            pred_trajectory.append(pred_np)
            current_pred_points = next_pred_points

    # 4. 制作动画
    print(f"Rendering visualization to {save_path}...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 确定边界
    all_pts = np.concatenate(gt_trajectory + pred_trajectory)
    min_b, max_b = all_pts.min(0) - 0.2, all_pts.max(0) + 0.2

    def update(frame):
        ax.clear()
        ax.set_xlim(min_b[0], max_b[0])
        ax.set_ylim(min_b[1], max_b[1])
        ax.set_zlim(min_b[2], max_b[2])
        ax.set_title(f"Rollout Frame {frame}/{frames}\nGreen: Physics (GT) | Red: Model Prediction")
        
        gt = gt_trajectory[frame]
        ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='green', alpha=0.4, label='Ground Truth')
        
        pred = pred_trajectory[frame]
        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='red', marker='x', s=20, label='Prediction')
        
        ax.legend()
        return ax,

    ani = FuncAnimation(fig, update, frames=frames, interval=100)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ani.save(save_path, writer='pillow', fps=10)
    print(f"Video saved successfully at: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    # 请确保该路径下有你训练好的模型
    ckpt = "checkpoints/model_epoch_50.pth"
    if os.path.exists(ckpt):
        make_rollout_video(ckpt)
    else:
        print(f"Error: Checkpoint {ckpt} not found. Please train the model first.")

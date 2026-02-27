import os
import sys
import numpy as np
import sapien.core as sapien
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.switch_backend('Agg')

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from se3_world_model.dataset import SapienSequenceDataset
from se3_world_model.model import SE3WorldModel

def sample_capsule_points(r: float, l: float, n: int) -> np.ndarray:
    points = []
    while len(points) < n:
        pt = np.random.uniform(low=[-l - r, -r, -r], high=[l + r, r, r])
        px = pt[0]
        if px < -l: dist = np.linalg.norm(pt - np.array([-l, 0, 0]))
        elif px > l: dist = np.linalg.norm(pt - np.array([l, 0, 0]))
        else: dist = np.linalg.norm(pt[1:])
        if dist <= r: points.append(pt)
    return np.array(points, dtype=np.float32)

def make_rollout_video(checkpoint_path: str, save_path: str = "assets/simulation.gif"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    HISTORY_LEN = 3 
    LATENT_DIM = 128
    NUM_POINTS = 64
    
    print("Loading dataset stats...")
    train_dataset = SapienSequenceDataset("data/sapien_train_seq.pt")
    
    pos_mean = train_dataset.pos_mean.to(device)
    pos_std = train_dataset.pos_std.to(device)
    vel_mean = train_dataset.vel_mean.to(device)
    vel_std = train_dataset.vel_std.to(device)
    # 【重构 1：新增提取力的全局均值与标准差】
    force_mean = train_dataset.force_mean.to(device)
    force_std = train_dataset.force_std.to(device)

    print(f"Stats loaded. Shared Std: {pos_std.mean():.4f}")

    print(f"Loading model from {checkpoint_path}...")
    model = SE3WorldModel(
        num_points=NUM_POINTS,
        latent_dim=LATENT_DIM, 
        num_global_vectors=1,
        context_dim=3,
        history_len=HISTORY_LEN 
    ).to(device)
    
    # 兼容 DDP 训练保存的带 'module.' 前缀的权重
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    print("Initializing SAPIEN simulation...")
    engine = sapien.Engine()
    engine.set_log_level("error")
    scene = engine.create_scene()
    sim_dt = 1 / 100.0
    scene.set_timestep(sim_dt)
    
    # 【重构 2：添加地面】
    scene.add_ground(altitude=0.0)

    gravity_np = np.array([0.0, -2.0, -9.0], dtype=np.float32)
    wind_np = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    total_force = gravity_np + wind_np

    builder = scene.create_actor_builder()
    builder.add_capsule_collision(radius=0.1, half_length=0.2)
    builder.set_mass_and_inertia(1.0, sapien.Pose(), np.array([0.1, 0.5, 0.8], dtype=np.float32))
    actor = builder.build(name="dynamic_object")
    
    # 【重构 3：抬高初始高度，让它在空中翻滚后撞击地面】
    actor.set_pose(sapien.Pose([0, 0, 2.0], [1, 0, 0, 0]))
    actor.set_velocity(np.array([0.0, 1.0, 0.0], dtype=np.float32)) 
    actor.set_angular_velocity(np.array([12.0, 8.0, 5.0], dtype=np.float32))

    canonical_points = sample_capsule_points(0.1, 0.2, NUM_POINTS)

    explicit_input = torch.tensor(gravity_np).float().to(device).view(1, 1, 3) 
    context_input = torch.tensor(wind_np).float().to(device).view(1, 3)

    history_x, history_v, history_f = [], [], []
    gt_trajectory, pred_trajectory = [], []

    print(f"Warming up for {HISTORY_LEN} steps to fill history buffer...")
    mat_prev = actor.get_pose().to_transformation_matrix()
    pts_prev = (mat_prev[:3, :3] @ canonical_points.T).T + mat_prev[:3, 3]

    for _ in range(HISTORY_LEN):
        total_contact_impulse = np.zeros(3, dtype=np.float32)
        sub_steps = 5
        
        for _ in range(sub_steps):
            actor.add_force_at_point(total_force, actor.get_pose().p)
            scene.step()
            
            # 提取碰撞冲量
            for contact in scene.get_contacts():
                if contact.actor0 == actor:
                    for pt in contact.points:
                        total_contact_impulse += pt.impulse
                elif contact.actor1 == actor:
                    for pt in contact.points:
                        total_contact_impulse -= pt.impulse
                        
        frame_dt = sub_steps * sim_dt
        avg_contact_force = total_contact_impulse / frame_dt
        point_forces = np.tile(avg_contact_force, (NUM_POINTS, 1))
            
        mat_curr = actor.get_pose().to_transformation_matrix()
        pts_curr = (mat_curr[:3, :3] @ canonical_points.T).T + mat_curr[:3, 3]
        velocity_raw = pts_curr - pts_prev
        
        pts_tensor = torch.tensor(pts_curr).float().to(device).unsqueeze(0) 
        vel_tensor = torch.tensor(velocity_raw).float().to(device).unsqueeze(0)
        force_tensor = torch.tensor(point_forces).float().to(device).unsqueeze(0)
        
        norm_x = (pts_tensor - pos_mean) / pos_std
        norm_v = (vel_tensor - vel_mean) / vel_std
        norm_f = (force_tensor - force_mean) / force_std # 归一化接触力
        
        history_x.append(norm_x)
        history_v.append(norm_v)
        history_f.append(norm_f)
        
        gt_trajectory.append(pts_curr)
        pts_prev = pts_curr

    # 【重构 4：拼接历史向量，新增 curr_f_hist】
    curr_x_hist = torch.stack(history_x, dim=1)
    curr_v_hist = torch.stack(history_v, dim=1)
    curr_f_hist = torch.stack(history_f, dim=1)
    
    last_x_real = (history_x[-1] * pos_std + pos_mean).cpu().numpy().squeeze(0)
    pred_trajectory.append(last_x_real) 

    frames = 60
    print("Running autoregressive rollout...")
    for _ in range(frames):
        total_contact_impulse = np.zeros(3, dtype=np.float32)
        sub_steps = 5
        
        # 跑 GT 仿真：获取 Ground Truth 轨迹以及下一步的“真实接触力”
        for _ in range(sub_steps):
            actor.add_force_at_point(total_force, actor.get_pose().p)
            scene.step()
            for contact in scene.get_contacts():
                if contact.actor0 == actor:
                    for pt in contact.points:
                        total_contact_impulse += pt.impulse
                elif contact.actor1 == actor:
                    for pt in contact.points:
                        total_contact_impulse -= pt.impulse
                        
        frame_dt = sub_steps * sim_dt
        avg_contact_force = total_contact_impulse / frame_dt
        point_forces = np.tile(avg_contact_force, (NUM_POINTS, 1))
        
        mat_new = actor.get_pose().to_transformation_matrix()
        pts_new = (mat_new[:3, :3] @ canonical_points.T).T + mat_new[:3, 3]
        gt_trajectory.append(pts_new) 
        
        # 将真实世界的物理接触力归一化，作为约束条件输给网络
        target_f_tensor = torch.tensor(point_forces).float().to(device).unsqueeze(0)
        target_f_norm = (target_f_tensor - force_mean) / force_std

        # 【重构 5：传入 curr_f_hist 执行推理预测】
        with torch.no_grad():
            pred_v_norm, _ = model(curr_x_hist, curr_v_hist, curr_f_hist, explicit_input, context_input)
            
            last_x_norm = curr_x_hist[:, -1] 
            pred_v_real = pred_v_norm * vel_std + vel_mean
            pred_x_norm = last_x_norm + pred_v_real / pos_std
            
            # 更新历史滑动窗口（自回归更新位置和速度，并压入真实的下一步受力）
            curr_x_hist = torch.cat([curr_x_hist[:, 1:], pred_x_norm.unsqueeze(1)], dim=1)
            curr_v_hist = torch.cat([curr_v_hist[:, 1:], pred_v_norm.unsqueeze(1)], dim=1)
            curr_f_hist = torch.cat([curr_f_hist[:, 1:], target_f_norm.unsqueeze(1)], dim=1)
            
            pred_real = pred_x_norm * pos_std + pos_mean
            pred_trajectory.append(pred_real.squeeze(0).cpu().numpy())

    print(f"Rendering video to {save_path}...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    min_len = min(len(gt_trajectory), len(pred_trajectory))
    gt_trajectory, pred_trajectory = gt_trajectory[:min_len], pred_trajectory[:min_len]
    
    all_pts = np.concatenate(gt_trajectory + pred_trajectory)
    min_b, max_b = all_pts.min(0) - 1.0, all_pts.max(0) + 1.0

    def update(frame):
        ax.clear()
        ax.set_xlim(min_b[0], max_b[0])
        ax.set_ylim(min_b[1], max_b[1])
        ax.set_zlim(min_b[2], max_b[2])
        # 锁定视角防止抖动
        ax.view_init(elev=20, azim=45) 
        ax.set_title(f"Rollout Frame {frame}/{min_len}\nGreen: Ground Truth | Red: Prediction")
        
        # 渲染地面 (简单的平面网格)
        xx, yy = np.meshgrid(np.linspace(min_b[0], max_b[0], 10), np.linspace(min_b[1], max_b[1], 10))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
        
        gt = gt_trajectory[frame]
        ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='green', alpha=0.4, label='GT')
        
        pred = pred_trajectory[frame]
        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='red', marker='x', alpha=0.8, label='Pred')
        ax.legend()
        return ax,

    ani = FuncAnimation(fig, update, frames=min_len, interval=100)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ani.save(save_path, writer='pillow', fps=10)
    print("Done!")

if __name__ == "__main__":
    # 使用你最新保存的最佳模型权重跑推理
    make_rollout_video("checkpoints/model_best.pth")

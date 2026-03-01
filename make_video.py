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
    train_dataset = SapienSequenceDataset("data/sapien_train_seq.h5", sub_seq_len=HISTORY_LEN+1)
    
    pos_mean = train_dataset.pos_mean.to(device)
    pos_std = train_dataset.pos_std.to(device)
    vel_std = train_dataset.vel_std.to(device)

    print(f"Stats loaded. Shared Std: {pos_std.mean():.4f}")

    print(f"Loading model from {checkpoint_path}...")
    model = SE3WorldModel(
        num_points=NUM_POINTS,
        latent_dim=LATENT_DIM, 
        num_global_vectors=2, # 【修复】：改为 2 (重力 + 风力)
        context_dim=1,        # 【修复】：改为 1 (占位符)
        history_len=HISTORY_LEN 
    ).to(device)
    
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
    
    scene.add_ground(altitude=0.0)

    gravity_np = np.array([0.0, 0.0, -9.8], dtype=np.float32)
    wind_np = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    total_force = gravity_np + wind_np

    builder = scene.create_actor_builder()
    builder.add_capsule_collision(radius=0.1, half_length=0.2)
    builder.set_mass_and_inertia(1.0, sapien.Pose(), np.array([0.1, 0.5, 0.8], dtype=np.float32))
    actor = builder.build(name="dynamic_object")
    
    actor.set_pose(sapien.Pose([0, 0, 2.0], [1, 0, 0, 0]))
    actor.set_velocity(np.array([0.0, 1.0, 0.0], dtype=np.float32)) 
    actor.set_angular_velocity(np.array([12.0, 8.0, 5.0], dtype=np.float32))

    canonical_points = sample_capsule_points(0.1, 0.2, NUM_POINTS)

    # 【核心修复】：将重力和风力合并为维度 [1, 2, 3] 的等变向量，替代之前的硬编码输入
    grav_tensor = torch.tensor(gravity_np).float().to(device).view(1, 1, 3) 
    wind_tensor = torch.tensor(wind_np).float().to(device).view(1, 1, 3)
    explicit_input = torch.cat([grav_tensor, wind_tensor], dim=1) # [1, 2, 3]
    dummy_ctx = torch.ones(1, 1, device=device) # [1, 1] 占位符

    history_x, history_v = [], []
    gt_trajectory, pred_trajectory = [], []

    print(f"Warming up for {HISTORY_LEN} steps to fill history buffer...")
    mat_prev = actor.get_pose().to_transformation_matrix()
    pts_prev = (mat_prev[:3, :3] @ canonical_points.T).T + mat_prev[:3, 3]

    for _ in range(HISTORY_LEN):
        sub_steps = 5
        
        for _ in range(sub_steps):
            actor.add_force_at_point(total_force, actor.get_pose().p)
            scene.step()
            
        mat_curr = actor.get_pose().to_transformation_matrix()
        pts_curr = (mat_curr[:3, :3] @ canonical_points.T).T + mat_curr[:3, 3]
        velocity_raw = pts_curr - pts_prev
        
        pts_tensor = torch.tensor(pts_curr).float().to(device).unsqueeze(0) 
        vel_tensor = torch.tensor(velocity_raw).float().to(device).unsqueeze(0)
        
        norm_x = (pts_tensor - pos_mean) / pos_std
        norm_v = vel_tensor / vel_std
        
        history_x.append(norm_x)
        history_v.append(norm_v)
        
        gt_trajectory.append(pts_curr)
        pts_prev = pts_curr

    curr_x_hist = torch.stack(history_x, dim=1)
    curr_v_hist = torch.stack(history_v, dim=1)
    
    last_x_real = (history_x[-1] * pos_std + pos_mean).cpu().numpy().squeeze(0)
    pred_trajectory.append(last_x_real) 

    frames = 60
    print("Running autoregressive rollout...")
    
    current_R = mat_curr[:3, :3]
    current_center_real = mat_curr[:3, 3]
    
    for _ in range(frames):
        sub_steps = 5
        
        for _ in range(sub_steps):
            actor.add_force_at_point(total_force, actor.get_pose().p)
            scene.step()
            
        mat_new = actor.get_pose().to_transformation_matrix()
        pts_new = (mat_new[:3, :3] @ canonical_points.T).T + mat_new[:3, 3]
        gt_trajectory.append(pts_new) 
        
        with torch.no_grad():
            # 【修复】：推理时传入更新后的 explicit_input 与 dummy_ctx
            pred_v_norm, _, next_v_cm_norm, R_pred = model(
                curr_x_hist, curr_v_hist, explicit_input, dummy_ctx, 
                vel_std=vel_std, pos_mean=pos_mean, pos_std=pos_std
            )
            
            v_cm_real = (next_v_cm_norm * vel_std).cpu().numpy().squeeze()
            R_mat = R_pred.cpu().numpy().squeeze()
            
            current_center_real += v_cm_real
            current_R = R_mat @ current_R
            
            pred_pts_real = (current_R @ canonical_points.T).T + current_center_real
            pred_trajectory.append(pred_pts_real)
            
            pred_x_norm = (torch.tensor(pred_pts_real).float().to(device).unsqueeze(0) - pos_mean) / pos_std
            
            pred_v_norm_step = (pred_x_norm - curr_x_hist[:, -1]) * (pos_std / vel_std)
            
            curr_x_hist = torch.cat([curr_x_hist[:, 1:], pred_x_norm.unsqueeze(1)], dim=1)
            curr_v_hist = torch.cat([curr_v_hist[:, 1:], pred_v_norm_step.unsqueeze(1)], dim=1)

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
        ax.view_init(elev=20, azim=45) 
        ax.set_title(f"Rollout Frame {frame}/{min_len}\nGreen: Ground Truth | Red: Prediction")
        
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
    make_rollout_video("checkpoints/model_best.pth")

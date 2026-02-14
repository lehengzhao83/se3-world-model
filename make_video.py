import os
import sys
import numpy as np
import sapien.core as sapien
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.switch_backend('Agg')
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from se3_world_model.dataset import SapienDataset
from se3_world_model.model import SE3WorldModel

# ... (sample_capsule_points same as before) ...
def sample_capsule_points(r, l, n):
    # (Copied from previous version to save space, please keep the implementation)
    points = []
    while len(points) < n:
        pt = np.random.uniform(low=[-l - r, -r, -r], high=[l + r, r, r])
        if pt[0] < -l: dist = np.linalg.norm(pt - np.array([-l, 0, 0]))
        elif pt[0] > l: dist = np.linalg.norm(pt - np.array([l, 0, 0]))
        else: dist = np.linalg.norm(pt[1:])
        if dist <= r: points.append(pt)
    return np.array(points, dtype=np.float32)

def make_rollout_video(checkpoint_path: str, save_path: str = "assets/simulation.gif"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 0. Load Stats
    print("Loading stats...")
    train_ds = SapienDataset("data/sapien_train.pt")
    pos_mean = train_ds.pos_mean.to(device)
    pos_std = train_ds.pos_std.to(device)
    vel_mean = train_ds.vel_mean.to(device)
    vel_std = train_ds.vel_std.to(device) # This is the key scaler!

    # 1. Load Model
    model = SE3WorldModel(num_points=64, latent_dim=64, num_global_vectors=1, context_dim=3).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
    model.eval()

    # 2. Setup SAPIEN
    engine = sapien.Engine()
    scene = engine.create_scene()
    scene.set_timestep(1/100.0)
    gravity_np = np.array([0.0, -2.0, -9.0], dtype=np.float32)
    wind_np = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    total_force = gravity_np + wind_np
    
    builder = scene.create_actor_builder()
    builder.add_capsule_collision(radius=0.1, half_length=0.2)
    builder.set_mass_and_inertia(1.0, sapien.Pose(), np.array([0.1, 0.1, 0.1], dtype=np.float32))
    actor = builder.build(name="dynamic_object")
    actor.set_pose(sapien.Pose([0,0,0], [1,0,0,0]))
    actor.set_velocity(np.array([0.0, 1.0, 2.0], dtype=np.float32)) # Initial velocity

    canonical_points = sample_capsule_points(0.1, 0.2, 64)

    # Warmup
    mat_prev = actor.get_pose().to_transformation_matrix()
    pts_prev = (mat_prev[:3, :3] @ canonical_points.T).T + mat_prev[:3, 3]
    for _ in range(5):
        actor.add_force_at_point(total_force, actor.get_pose().p)
        scene.step()
    mat_curr = actor.get_pose().to_transformation_matrix()
    pts_curr = (mat_curr[:3, :3] @ canonical_points.T).T + mat_curr[:3, 3]
    
    # Initial State (Real World)
    current_pos_real = torch.tensor(pts_curr).float().to(device).unsqueeze(0)
    current_vel_real = torch.tensor(pts_curr - pts_prev).float().to(device).unsqueeze(0)
    
    explicit = torch.tensor(gravity_np).float().to(device).view(1, 1, 3)
    context = torch.tensor(wind_np).float().to(device).view(1, 3)

    frames = 60
    gt_traj, pred_traj = [], []

    print("Running rollout...")
    for _ in range(frames):
        # A. GT
        gt_traj.append(pts_curr)
        for _ in range(5):
            actor.add_force_at_point(total_force, actor.get_pose().p)
            scene.step()
        mat_new = actor.get_pose().to_transformation_matrix()
        pts_new = (mat_new[:3, :3] @ canonical_points.T).T + mat_new[:3, 3]
        pts_curr = pts_new

        # B. Pred
        with torch.no_grad():
            # 1. Normalize Inputs
            x_norm = (current_pos_real - pos_mean) / pos_std
            v_norm = (current_vel_real - vel_mean) / vel_std
            
            # 2. Predict Velocity (Normalized)
            pred_v_norm, _ = model(x_norm, v_norm, explicit, context)
            
            # 3. Denormalize Velocity (Scale it back up!)
            pred_v_real = pred_v_norm * vel_std + vel_mean
            
            # 4. Integrate Position
            next_pos_real = current_pos_real + pred_v_real
            
            # Store & Update
            pred_traj.append(next_pos_real.squeeze(0).cpu().numpy())
            current_pos_real = next_pos_real
            current_vel_real = pred_v_real # Update velocity for next step

    # Rendering
    print(f"Rendering to {save_path}...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    all_pts = np.concatenate(gt_traj + pred_traj)
    min_b, max_b = all_pts.min(0) - 1, all_pts.max(0) + 1

    def update(frame):
        ax.clear()
        ax.set_xlim(min_b[0], max_b[0]); ax.set_ylim(min_b[1], max_b[1]); ax.set_zlim(min_b[2], max_b[2])
        ax.set_title(f"Frame {frame}/{frames}\nGreen: GT | Red: Pred (Vel-Based)")
        gt = gt_traj[frame]
        ax.scatter(gt[:,0], gt[:,1], gt[:,2], c='green', alpha=0.4)
        pred = pred_traj[frame]
        ax.scatter(pred[:,0], pred[:,1], pred[:,2], c='red', marker='x')
        return ax,

    ani = FuncAnimation(fig, update, frames=frames, interval=100)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ani.save(save_path, writer='pillow', fps=10)
    print("Done!")

if __name__ == "__main__":
    make_rollout_video("checkpoints/model_epoch_50.pth")

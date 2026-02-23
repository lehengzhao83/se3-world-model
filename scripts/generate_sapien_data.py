import os
import numpy as np
import sapien.core as sapien
import torch
from tqdm import tqdm

def sample_capsule_points(r: float, l: float, n: int) -> np.ndarray:
    points = []
    while len(points) < n:
        pt = np.random.uniform(low=[-l - r, -r, -r], high=[l + r, r, r])
        if pt[0] < -l: dist = np.linalg.norm(pt - np.array([-l, 0, 0]))
        elif pt[0] > l: dist = np.linalg.norm(pt - np.array([l, 0, 0]))
        else: dist = np.linalg.norm(pt[1:])
        if dist <= r: points.append(pt)
    return np.array(points, dtype=np.float32)

def generate_trajectory_dataset(
    num_trajectories: int = 10000, 
    traj_len: int = 60, 
    save_path: str = "data/sapien_train_seq.pt"
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Generating {num_trajectories} trajectories of length {traj_len}...")

    engine = sapien.Engine()
    engine.set_log_level("error")
    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)
    
    # 彻底移除地面，避免打破模型的 SE(3) 平移不变性

    canonical_points = sample_capsule_points(0.1, 0.2, 64)
    all_x, all_v, all_explicit, all_context = [], [], [], []

    builder = scene.create_actor_builder()
    builder.add_capsule_collision(radius=0.1, half_length=0.2)
    
    # === 核心物理：设置不对称的转动惯量 (触发贾尼别科夫翻转) ===
    builder.set_mass_and_inertia(1.0, sapien.Pose(), np.array([0.1, 0.5, 0.8], dtype=np.float32))
    actor = builder.build(name="dynamic_object")

    for _ in tqdm(range(num_trajectories)):
        gravity_acc = np.random.randn(3).astype(np.float32)
        gravity_acc = gravity_acc / (np.linalg.norm(gravity_acc) + 1e-6) * 9.8
        wind_force = np.random.randn(3).astype(np.float32) * 3.0 
        total_force = gravity_acc + wind_force

        pos = np.random.randn(3).astype(np.float32) * 0.5
        q = np.random.randn(4).astype(np.float32)
        q /= (np.linalg.norm(q) + 1e-6)
        actor.set_pose(sapien.Pose(pos, q))
        
        actor.set_velocity(np.random.randn(3).astype(np.float32) * 3.0)
        
        # === 核心物理：赋予强烈的初始三轴角速度 ===
        w_init = np.random.randn(3).astype(np.float32) * 12.0
        actor.set_angular_velocity(w_init)
        
        traj_x, traj_v = [], []
        
        mat = actor.get_pose().to_transformation_matrix()
        pts_prev = (mat[:3, :3] @ canonical_points.T).T + mat[:3, 3]
        
        for _ in range(5):
            actor.add_force_at_point(total_force, actor.get_pose().p)
            scene.step()

        for t in range(traj_len):
            mat = actor.get_pose().to_transformation_matrix()
            pts_curr = (mat[:3, :3] @ canonical_points.T).T + mat[:3, 3]
            vel = pts_curr - pts_prev
            
            traj_x.append(pts_curr)
            traj_v.append(vel)
            pts_prev = pts_curr
            
            for _ in range(5):
                actor.add_force_at_point(total_force, actor.get_pose().p)
                scene.step()
        
        all_x.append(np.stack(traj_x))
        all_v.append(np.stack(traj_v))
        all_explicit.append(np.tile(gravity_acc.reshape(1, 1, 3), (traj_len, 1, 1)))
        all_context.append(np.tile(wind_force.reshape(1, 3), (traj_len, 1)))

    data_dict = {
        "x": torch.tensor(np.stack(all_x)),
        "v": torch.tensor(np.stack(all_v)),
        "explicit": torch.tensor(np.stack(all_explicit)),
        "context": torch.tensor(np.stack(all_context))
    }
    torch.save(data_dict, save_path)

if __name__ == "__main__":
    generate_trajectory_dataset(10000, 60, "data/sapien_train_seq.pt")
    generate_trajectory_dataset(200, 60, "data/sapien_val_seq.pt")

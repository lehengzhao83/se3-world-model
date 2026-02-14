import os
import numpy as np
import sapien.core as sapien
import torch
from tqdm import tqdm

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

def generate_trajectory_dataset(
    num_trajectories: int = 2000, 
    seq_len: int = 10,  # [核心修改] 每个样本是一条长为 10 的轨迹
    save_path: str = "data/sapien_train_seq.pt"
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Generating {num_trajectories} trajectories (Length={seq_len})...")

    engine = sapien.Engine()
    engine.set_log_level("error")
    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    canonical_points = sample_capsule_points(0.1, 0.2, 64)
    
    # 存储容器：List of sequences
    # 最终形状将是 [N_traj, Seq_Len, ...]
    data_store = {
        "x": [], 
        "v": [], 
        "explicit": [], 
        "context": []
    }

    builder = scene.create_actor_builder()
    builder.add_capsule_collision(radius=0.1, half_length=0.2)
    builder.set_mass_and_inertia(1.0, sapien.Pose(), np.array([0.1, 0.1, 0.1], dtype=np.float32))
    actor = builder.build(name="dynamic_object")

    for _ in tqdm(range(num_trajectories)):
        # 1. 随机力场
        gravity_acc = np.random.randn(3).astype(np.float32)
        gravity_acc = gravity_acc / (np.linalg.norm(gravity_acc) + 1e-6) * 9.8
        wind_force = np.random.randn(3).astype(np.float32) * 2.0
        total_force = gravity_acc + wind_force

        # 2. 随机初始状态
        pos = np.random.randn(3).astype(np.float32) * 0.5
        q = np.random.randn(4).astype(np.float32)
        q /= np.linalg.norm(q)
        actor.set_pose(sapien.Pose(pos, q))
        
        # 关键：赋予随机初速度，否则很难学到动态变化
        v_init = np.random.randn(3).astype(np.float32) * 2.0
        actor.set_velocity(v_init)
        
        # 3. 预热 (Warmup) 获取 t=0 的状态
        # T-1
        mat = actor.get_pose().to_transformation_matrix()
        pts_prev = (mat[:3, :3] @ canonical_points.T).T + mat[:3, 3]
        
        # Sim step
        for _ in range(5):
            actor.add_force_at_point(total_force, actor.get_pose().p)
            scene.step()

        # 4. 生成序列
        traj_x = []
        traj_v = []
        
        # 我们需要收集 seq_len 个连续帧
        for t in range(seq_len):
            # Capture Current (T)
            mat = actor.get_pose().to_transformation_matrix()
            pts_curr = (mat[:3, :3] @ canonical_points.T).T + mat[:3, 3]
            
            # Compute Velocity (T - (T-1))
            vel = pts_curr - pts_prev
            
            traj_x.append(pts_curr)
            traj_v.append(vel)
            
            # Update Prev
            pts_prev = pts_curr
            
            # Sim next step
            for _ in range(5):
                actor.add_force_at_point(total_force, actor.get_pose().p)
                scene.step()
        
        # Stack trajectory
        data_store["x"].append(np.stack(traj_x))       # [Seq_Len, N, 3]
        data_store["v"].append(np.stack(traj_v))       # [Seq_Len, N, 3]
        
        # Force is constant for the whole trajectory
        # Expand to [Seq_Len, 1, 3]
        data_store["explicit"].append(np.tile(gravity_acc.reshape(1, 1, 3), (seq_len, 1, 1)))
        data_store["context"].append(np.tile(wind_force.reshape(1, 3), (seq_len, 1)))

    # Convert to Tensor
    for k in data_store:
        data_store[k] = torch.tensor(np.array(data_store[k]))

    torch.save(data_store, save_path)
    print(f"Saved sequential dataset to {save_path}")

if __name__ == "__main__":
    # 2000 条轨迹，每条 20 帧，相当于 40000 个单步样本，但包含了时序信息
    generate_trajectory_dataset(2000, 20, "data/sapien_train_seq.pt") 
    generate_trajectory_dataset(200, 20, "data/sapien_val_seq.pt")

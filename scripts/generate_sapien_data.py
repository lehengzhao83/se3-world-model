import os
import numpy as np
import sapien.core as sapien
import h5py
from tqdm import tqdm

def sample_capsule_points(r: float, l: float, n: int) -> np.ndarray:
    """
    在胶囊体（Capsule）内部均匀采样 n 个点，作为代表该刚体的点云。
    """
    points = []
    while len(points) < n:
        pt = np.random.uniform(low=[-l - r, -r, -r], high=[l + r, r, r])
        
        if pt[0] < -l: 
            dist = np.linalg.norm(pt - np.array([-l, 0, 0]))
        elif pt[0] > l: 
            dist = np.linalg.norm(pt - np.array([l, 0, 0]))
        else: 
            dist = np.linalg.norm(pt[1:])
            
        if dist <= r: 
            points.append(pt)
            
    return np.array(points, dtype=np.float32)

def generate_trajectory_dataset(
    num_trajectories: int = 1000000, 
    traj_len: int = 60, 
    save_path: str = "data/sapien_train_seq.h5" # 修改为 HDF5 格式
) -> None:
    """
    使用 SAPIEN 物理引擎生成刚体运动轨迹数据集 (HDF5 流式写入版本)。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Generating {num_trajectories} trajectories of length {traj_len}...")
    print(f"Data will be streamed to {save_path}")

    # === 1. 初始化 SAPIEN 物理引擎 ===
    engine = sapien.Engine()
    engine.set_log_level("error")
    scene = engine.create_scene()
    
    sim_dt = 1 / 100.0
    scene.set_timestep(sim_dt)
    scene.add_ground(altitude=0.0)

    # === 2. 准备刚体规范点云 ===
    num_points = 64
    canonical_points = sample_capsule_points(0.1, 0.2, num_points)

    # === 3. 构建刚体 Actor ===
    builder = scene.create_actor_builder()
    builder.add_capsule_collision(radius=0.1, half_length=0.2)
    builder.set_mass_and_inertia(1.0, sapien.Pose(), np.array([0.1, 0.5, 0.8], dtype=np.float32))
    actor = builder.build(name="dynamic_object")

    # === 核心重构：创建 HDF5 文件并初始化流式数据集 ===
    with h5py.File(save_path, 'w') as f:
        # 预分配磁盘空间，开启 chunking (块存储) 以极大优化 DDP 训练时的随机采样读取速度
        ds_x = f.create_dataset("x", (num_trajectories, traj_len, num_points, 3), dtype='f4', chunks=(1, traj_len, num_points, 3))
        ds_v = f.create_dataset("v", (num_trajectories, traj_len, num_points, 3), dtype='f4', chunks=(1, traj_len, num_points, 3))
        ds_f = f.create_dataset("force", (num_trajectories, traj_len, num_points, 3), dtype='f4', chunks=(1, traj_len, num_points, 3))
        ds_expl = f.create_dataset("explicit", (num_trajectories, traj_len, 1, 3), dtype='f4')
        ds_ctx = f.create_dataset("context", (num_trajectories, traj_len, 3), dtype='f4')

        # 在线计算均值和方差的累加器
        sum_pos = np.zeros(3, dtype=np.float64)
        sum_vel = np.zeros(3, dtype=np.float64)
        sum_force = np.zeros(3, dtype=np.float64)
        
        # 标量标准差所需的标量累加器
        sum_vel_scalar = 0.0
        sum_force_scalar = 0.0
        sq_sum_vel_scalar = 0.0
        sq_sum_force_scalar = 0.0

        # === 4. 开始生成轨迹 ===
        for i in tqdm(range(num_trajectories)):
            # --- 4.1 随机化物理环境参数 ---
            gravity_acc = np.random.randn(3).astype(np.float32)
            gravity_acc = gravity_acc / (np.linalg.norm(gravity_acc) + 1e-6) * 9.8 
            wind_force = np.random.randn(3).astype(np.float32) * 3.0 
            total_force = gravity_acc + wind_force

            # --- 4.2 随机化初始状态 ---
            pos = np.random.randn(3).astype(np.float32) * 0.5
            pos[2] = np.random.uniform(1.0, 2.5) # 半空生成
            
            q = np.random.randn(4).astype(np.float32)
            q /= (np.linalg.norm(q) + 1e-6) 
            
            actor.set_pose(sapien.Pose(pos, q))
            actor.set_velocity(np.random.randn(3).astype(np.float32) * 3.0)
            
            w_init = np.random.randn(3).astype(np.float32) * 12.0
            actor.set_angular_velocity(w_init)
            
            traj_x, traj_v, traj_f = [], [], []
            
            # --- 4.3 物理预热 ---
            mat = actor.get_pose().to_transformation_matrix()
            pts_prev = (mat[:3, :3] @ canonical_points.T).T + mat[:3, 3]
            
            for _ in range(5):
                actor.add_force_at_point(total_force, actor.get_pose().p)
                scene.step()

            # --- 4.4 记录轨迹 ---
            for t in range(traj_len):
                mat = actor.get_pose().to_transformation_matrix()
                pts_curr = (mat[:3, :3] @ canonical_points.T).T + mat[:3, 3]
                vel = pts_curr - pts_prev
                
                traj_x.append(pts_curr)
                traj_v.append(vel)
                pts_prev = pts_curr
                
                total_contact_impulse = np.zeros(3, dtype=np.float32)
                sub_steps = 5
                
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
                point_forces = np.tile(avg_contact_force, (num_points, 1))
                traj_f.append(point_forces)
            
            arr_x = np.stack(traj_x)
            arr_v = np.stack(traj_v)
            arr_f = np.stack(traj_f)
            
            # --- 4.5 写入硬盘并释放内存 ---
            ds_x[i] = arr_x
            ds_v[i] = arr_v
            ds_f[i] = arr_f
            ds_expl[i] = np.tile(gravity_acc.reshape(1, 1, 3), (traj_len, 1, 1))
            ds_ctx[i] = np.tile(wind_force.reshape(1, 3), (traj_len, 1))

            # --- 4.6 在线累加统计量 ---
            sum_pos += arr_x.mean(axis=(0, 1))     # [3,]
            sum_vel += arr_v.mean(axis=(0, 1))     # [3,]
            sum_force += arr_f.mean(axis=(0, 1))   # [3,]
            
            sum_vel_scalar += arr_v.mean()
            sum_force_scalar += arr_f.mean()
            sq_sum_vel_scalar += (arr_v ** 2).mean()
            sq_sum_force_scalar += (arr_f ** 2).mean()

        # === 5. 计算最终的全局物理统计量并存入 HDF5 属性中 ===
        print("Calculating dataset statistics...")
        f.attrs['pos_mean'] = sum_pos / num_trajectories
        f.attrs['vel_mean'] = sum_vel / num_trajectories
        f.attrs['force_mean'] = sum_force / num_trajectories
        
        vel_mean_scalar = sum_vel_scalar / num_trajectories
        force_mean_scalar = sum_force_scalar / num_trajectories
        
        # 共享标量方差 (严格维持 SE(3) 等变性度量)
        f.attrs['vel_std'] = np.sqrt(sq_sum_vel_scalar / num_trajectories - vel_mean_scalar**2)
        f.attrs['force_std'] = np.sqrt(sq_sum_force_scalar / num_trajectories - force_mean_scalar**2)
        
        print("Dataset generation completed successfully!")

if __name__ == "__main__":
    # 生成 100000 条训练数据 (HDF5)
    generate_trajectory_dataset(100000, 60, "data/sapien_train_seq.h5")
    # 生成 2000 条验证数据 (HDF5)
    generate_trajectory_dataset(2000, 60, "data/sapien_val_seq.h5")

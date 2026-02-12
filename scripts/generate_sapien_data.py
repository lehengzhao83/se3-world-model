import os
import numpy as np
import sapien.core as sapien
import torch
from tqdm import tqdm

def sample_capsule_points(r: float, l: float, n: int) -> np.ndarray:
    """在胶囊体表面/内部均匀采样点云"""
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

def generate_dataset(num_samples: int = 10000, save_path: str = "data/sapien_train.pt") -> None:
    # 确保文件夹存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"正在初始化 SAPIEN 引擎 (物理模拟模式)...目标：{num_samples} 条")

    # 1. 初始化引擎
    engine = sapien.Engine()
    engine.set_log_level("error")
    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    # 2. 预计算标准点云
    canonical_points = sample_capsule_points(0.1, 0.2, 64)
    all_x_t, all_explicit, all_context, all_x_next = [], [], [], []

    # 3. 创建物体
    builder = scene.create_actor_builder()
    builder.add_capsule_collision(radius=0.1, half_length=0.2)
    # 显式传递参数：质量1.0，单位姿态，惯性张量
    builder.set_mass_and_inertia(
        1.0, 
        sapien.Pose([0, 0, 0], [1, 0, 0, 0]), 
        np.array([0.1, 0.1, 0.1], dtype=np.float32)
    )
    actor = builder.build(name="dynamic_object")

    # 4. 开始生成
    for _ in tqdm(range(num_samples)):
        # 随机重力加速度 (a) 和 风力 (f)
        gravity_acc = np.random.randn(3).astype(np.float32)
        gravity_acc = gravity_acc / (np.linalg.norm(gravity_acc) + 1e-6) * 9.8
        wind_force = np.random.randn(3).astype(np.float32) * 2.0
        
        # 合力 (因为 m=1.0, 所以合力数值上等于合加速度)
        total_force = gravity_acc + wind_force

        # 随机重置状态
        pos = np.random.randn(3).astype(np.float32) * 0.5
        q = np.random.randn(4).astype(np.float32)
        q /= (np.linalg.norm(q) + 1e-6)
        actor.set_pose(sapien.Pose(pos, q))
        actor.set_velocity([0, 0, 0])
        actor.set_angular_velocity([0, 0, 0])

        # 记录 t 时刻点云
        mat_t = actor.get_pose().to_transformation_matrix()
        pts_t = (mat_t[:3, :3] @ canonical_points.T).T + mat_t[:3, 3]

        # 5. 物理模拟步进 (t -> t+1)
        for _ in range(5):
            # 获取物体当前位置，作为施力点（即中心点）
            current_pos = actor.get_pose().p
            # 使用报错建议的 API：add_force_at_point(力向量, 作用点)
            actor.add_force_at_point(total_force, current_pos)
            scene.step()

        # 记录 t+1 时刻点云
        mat_next = actor.get_pose().to_transformation_matrix()
        pts_next = (mat_next[:3, :3] @ canonical_points.T).T + mat_next[:3, 3]

        all_x_t.append(pts_t)
        all_explicit.append(gravity_acc.reshape(1, 3))
        all_context.append(wind_force)
        all_x_next.append(pts_next)

    # 6. 保存数据
    torch.save({
        "x_t": torch.tensor(np.array(all_x_t)),
        "explicit": torch.tensor(np.array(all_explicit)),
        "context": torch.tensor(np.array(all_context)),
        "x_next": torch.tensor(np.array(all_x_next))
    }, save_path)
    print(f"数据已保存至 {save_path}")

if __name__ == "__main__":
    generate_dataset(10000, "data/sapien_train.pt")
    generate_dataset(1000, "data/sapien_val.pt")

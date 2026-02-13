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
    all_x_t, all_v_t, all_explicit, all_context, all_x_next = [], [], [], [], []

    # 3. 创建物体
    builder = scene.create_actor_builder()
    builder.add_capsule_collision(radius=0.1, half_length=0.2)
    builder.set_mass_and_inertia(
        1.0,
        sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
        np.array([0.1, 0.1, 0.1], dtype=np.float32)
    )
    actor = builder.build(name="dynamic_object")

    # 4. 开始生成
    for _ in tqdm(range(num_samples)):
        # 随机环境力
        gravity_acc = np.random.randn(3).astype(np.float32)
        gravity_acc = gravity_acc / (np.linalg.norm(gravity_acc) + 1e-6) * 9.8
        wind_force = np.random.randn(3).astype(np.float32) * 2.0
        total_force = gravity_acc + wind_force

        # 随机初始化状态
        pos = np.random.randn(3).astype(np.float32) * 0.5
        q = np.random.randn(4).astype(np.float32)
        q /= (np.linalg.norm(q) + 1e-6)
        actor.set_pose(sapien.Pose(pos, q))
        actor.set_velocity([0, 0, 0])
        actor.set_angular_velocity([0, 0, 0])

        # === 关键修改：两段式模拟以获取速度 ===

        # Phase 1: 模拟 t-1 -> t (Warmup)
        for _ in range(5):
            current_pos = actor.get_pose().p
            actor.add_force_at_point(total_force, current_pos)
            scene.step()

        # 记录 t 时刻状态 (Input)
        mat_t = actor.get_pose().to_transformation_matrix()
        pts_t = (mat_t[:3, :3] @ canonical_points.T).T + mat_t[:3, 3]

        # 记录上一帧状态用于计算速度 (Pre-Input)
        # 注意：这里我们实际上不需要 pts_prev 的绝对坐标，只需要算出 v
        # 但为了严谨，我们应该记录 warmup 之前的状态吗？
        # 不，最好的方式是：Warmup 5步 -> 记录 Prev -> Sim 5步 -> 记录 Curr -> Sim 5步 -> 记录 Next
        # 为了简化且保持物理连续性，我们直接利用 SAPIEN 的速度，或者再跑一段。
        # 修正逻辑：
        # Reset -> 记录 pts_prev -> Sim 5步 -> 记录 pts_t -> Sim 5步 -> 记录 pts_next
        # 这样 v = pts_t - pts_prev 是完全真实的物理位移速度

        # 重来一次逻辑：
        # 1. Reset
        # 2. Capture T-1
        mat_prev = actor.get_pose().to_transformation_matrix()
        pts_prev = (mat_prev[:3, :3] @ canonical_points.T).T + mat_prev[:3, 3]

        # 3. Sim 5 steps (T-1 -> T)
        for _ in range(5):
            actor.add_force_at_point(total_force, actor.get_pose().p)
            scene.step()

        # 4. Capture T (Input)
        mat_t = actor.get_pose().to_transformation_matrix()
        pts_t = (mat_t[:3, :3] @ canonical_points.T).T + mat_t[:3, 3]

        # 5. Compute Velocity (Input)
        velocity = pts_t - pts_prev

        # 6. Sim 5 steps (T -> T+1)
        for _ in range(5):
            actor.add_force_at_point(total_force, actor.get_pose().p)
            scene.step()

        # 7. Capture T+1 (Target)
        mat_next = actor.get_pose().to_transformation_matrix()
        pts_next = (mat_next[:3, :3] @ canonical_points.T).T + mat_next[:3, 3]

        all_x_t.append(pts_t)
        all_v_t.append(velocity)  # 新增：速度场
        all_explicit.append(gravity_acc.reshape(1, 3))
        all_context.append(wind_force)
        all_x_next.append(pts_next)

    # 6. 保存数据
    torch.save({
        "x_t": torch.tensor(np.array(all_x_t)),
        "v_t": torch.tensor(np.array(all_v_t)),  # 保存速度
        "explicit": torch.tensor(np.array(all_explicit)),
        "context": torch.tensor(np.array(all_context)),
        "x_next": torch.tensor(np.array(all_x_next))
    }, save_path)
    print(f"数据已保存至 {save_path} (含速度信息)")


if __name__ == "__main__":
    generate_dataset(10000, "data/sapien_train.pt")
    generate_dataset(1000, "data/sapien_val.pt")

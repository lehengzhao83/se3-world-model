import os
import sys
import numpy as np
import sapien.core as sapien
import torch
from tqdm import tqdm

def sample_capsule_points(r: float, l: float, n: int) -> np.ndarray:
    """在胶囊体表面/内部均匀采样点云"""
    points = []
    while len(points) < n:
        # 在包围盒内采样: [-l-r, l+r] x [-r, r] x [-r, r]
        # SAPIEN 胶囊默认沿 X 轴
        pt = np.random.uniform(low=[-l - r, -r, -r], high=[l + r, r, r])
        
        # 判断是否在胶囊内
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

def generate_dataset(num_samples=10000, save_path="data/sapien_train.pt"):
    print(f"正在初始化 SAPIEN 引擎，准备生成 {num_samples} 条数据...")
    
    # 初始化 SAPIEN (无渲染模式，纯物理)
    engine = sapien.Engine()
    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    # 预计算胶囊体的标准点云 (Canonical Point Cloud)
    # 半径 0.1, 半长 0.2
    canonical_points = sample_capsule_points(0.1, 0.2, 64)

    # 数据容器
    all_x_t = []
    all_explicit = []
    all_context = []
    all_x_next = []

    # 创建一个用于模拟的动态物体
    builder = scene.create_actor_builder()
    builder.add_capsule_collision(radius=0.1, half_length=0.2)
    builder.set_mass_and_inertia(mass=1.0, mass_center=[0,0,0], inertia=[0.1, 0.1, 0.1])
    actor = builder.build(name="dynamic_object")

    # 开始循环生成
    for _ in tqdm(range(num_samples)):
        # 1. 随机化环境力
        # 显式力：重力 (随机方向，模拟不同环境)
        # 正常重力是 (0, 0, -9.8)，我们随机化它，强迫模型学习方向
        gravity = np.random.randn(3)
        gravity = gravity / np.linalg.norm(gravity) * 9.8
        scene.set_gravity(gravity)

        # 隐式力：风力 (随机向量，不可见，作为 Context 输入)
        wind = np.random.randn(3) * 2.0 

        # 2. 重置物体状态
        # 随机位置和姿态
        pos = np.random.randn(3) * 0.5
        q = np.random.randn(4)
        q /= np.linalg.norm(q)
        actor.set_pose(sapien.Pose(pos, q))
        
        # 随机初速度
        actor.set_velocity(np.random.randn(3) * 0.5)
        actor.set_angular_velocity(np.random.randn(3) * 0.5)

        # 3. 获取 t 时刻状态 (Input)
        pose_t = actor.get_pose()
        mat_t = pose_t.to_transformation_matrix()
        # 将标准点云变换到当前位姿: R*p + t
        pts_t = (mat_t[:3, :3] @ canonical_points.T).T + mat_t[:3, 3]

        # 4. 物理模拟 (t -> t+1)
        # 模拟 5 个物理步长
        for _ in range(5):
            actor.add_force_accum(wind) # 施加风力
            scene.step()

        # 5. 获取 t+1 时刻状态 (Target)
        pose_next = actor.get_pose()
        mat_next = pose_next.to_transformation_matrix()
        pts_next = (mat_next[:3, :3] @ canonical_points.T).T + mat_next[:3, 3]

        # 6. 收集数据
        all_x_t.append(pts_t)
        all_explicit.append(gravity.reshape(1, 3)) # Explicit: [1, 3]
        all_context.append(wind)                   # Implicit: [3]
        all_x_next.append(pts_next)

    # 转换为 Tensor 并保存
    data_dict = {
        "x_t": torch.tensor(np.array(all_x_t), dtype=torch.float32),
        "explicit": torch.tensor(np.array(all_explicit), dtype=torch.float32),
        "context": torch.tensor(np.array(all_context), dtype=torch.float32),
        "x_next": torch.tensor(np.array(all_x_next), dtype=torch.float32)
    }
    
    print(f"保存数据到 {save_path} ...")
    torch.save(data_dict, save_path)
    print("完成！")

if __name__ == "__main__":
    # 生成训练集 (10000条)
    generate_dataset(10000, "data/sapien_train.pt")
    # 生成验证集 (1000条)
    generate_dataset(1000, "data/sapien_val.pt")

import os
import numpy as np
import sapien.core as sapien
import torch
from tqdm import tqdm

def sample_capsule_points(r: float, l: float, n: int) -> np.ndarray:
    """
    在胶囊体（Capsule）内部均匀采样 n 个点，作为代表该刚体的点云。
    胶囊体由一个圆柱体（长 2*l）和两端的半球（半径 r）组成。
    
    Args:
        r: 胶囊体两端半球的半径
        l: 胶囊体中间圆柱部分的半长 (half_length)
        n: 需要采样的点数
    Returns:
        形状为 (n, 3) 的 NumPy 数组，包含局部坐标系下的 3D 点云。
    """
    points = []
    # 使用拒绝采样（Rejection Sampling）法
    while len(points) < n:
        # 在包围该胶囊体的长方体 Bounding Box 内随机生成一个点
        pt = np.random.uniform(low=[-l - r, -r, -r], high=[l + r, r, r])
        
        # 计算该点到胶囊体中心轴（x 轴上从 -l 到 l 的线段）的最短距离
        if pt[0] < -l: 
            # 如果点在左侧半球区域外，计算到左端点 (-l, 0, 0) 的距离
            dist = np.linalg.norm(pt - np.array([-l, 0, 0]))
        elif pt[0] > l: 
            # 如果点在右侧半球区域外，计算到右端点 (l, 0, 0) 的距离
            dist = np.linalg.norm(pt - np.array([l, 0, 0]))
        else: 
            # 如果点在中间圆柱区域内，计算到 x 轴的垂直距离（即 yz 平面的模长）
            dist = np.linalg.norm(pt[1:])
            
        # 如果距离小于等于半径 r，说明点在胶囊体内部，接受该点
        if dist <= r: 
            points.append(pt)
            
    return np.array(points, dtype=np.float32)

def generate_trajectory_dataset(
    num_trajectories: int = 10000, 
    traj_len: int = 60, 
    save_path: str = "data/sapien_train_seq.pt"
) -> None:
    """
    使用 SAPIEN 物理引擎生成刚体运动轨迹数据集。
    
    Args:
        num_trajectories: 需要生成的独立轨迹（场景）总数
        traj_len: 每条轨迹记录的时间步数
        save_path: 数据集保存路径
    """
    # 确保保存数据的目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Generating {num_trajectories} trajectories of length {traj_len}...")

    # === 1. 初始化 SAPIEN 物理引擎 ===
    engine = sapien.Engine()
    engine.set_log_level("error") # 隐藏不必要的底层物理日志
    scene = engine.create_scene() # 创建物理场景
    # 设置物理仿真的基础时间步长。值越小，仿真越精确但越慢。这里设置为 100Hz。
    scene.set_timestep(1 / 100.0)
    
    # 【注】：这里特意没有添加地面 (Ground)。
    # 彻底移除地面，避免碰撞产生的边界条件打破模型期望学习的 SE(3) 平移和旋转不变性。

    # === 2. 准备刚体规范点云 ===
    # 生成胶囊体在局部坐标系（Canonical Space）下的点云模板
    canonical_points = sample_capsule_points(0.1, 0.2, 64)
    
    # 初始化用于存储所有轨迹数据的列表
    all_x, all_v, all_explicit, all_context = [], [], [], []

    # === 3. 构建刚体 Actor ===
    builder = scene.create_actor_builder()
    # 添加一个胶囊体碰撞形状，这决定了物体的几何属性和碰撞边界
    builder.add_capsule_collision(radius=0.1, half_length=0.2)
    
    # === 核心物理：设置不对称的转动惯量 (触发贾尼别科夫翻转 Dzhanibekov Effect) ===
    # 质量设为 1.0，质心位姿为默认。
    # 转动惯量对角线设置为 [0.1, 0.5, 0.8]，这意味着物体在三个主轴上的惯性完全不同。
    # 这种配置下，绕中间惯量主轴（y轴）的旋转是不稳定的，会产生复杂的周期性翻滚。
    # 这是检验 SE(3) 等变模型预测复杂非线性旋转能力的绝佳测试。
    builder.set_mass_and_inertia(1.0, sapien.Pose(), np.array([0.1, 0.5, 0.8], dtype=np.float32))
    
    # 构建为动态物体（Dynamic Object），受物理引擎控制
    actor = builder.build(name="dynamic_object")

    # === 4. 开始生成轨迹 ===
    for _ in tqdm(range(num_trajectories)):
        
        # --- 4.1 随机化物理环境参数 ---
        # 随机生成全局重力加速度方向和大小（显式等变向量 Type-1）
        gravity_acc = np.random.randn(3).astype(np.float32)
        gravity_acc = gravity_acc / (np.linalg.norm(gravity_acc) + 1e-6) * 9.8 # 归一化后乘以 9.8 模拟地球重力大小
        
        # 随机生成环境风力（隐式上下文标量特征 Type-0，它会打破空间的各向同性）
        wind_force = np.random.randn(3).astype(np.float32) * 3.0 
        
        # 物体受到的恒定外力总和
        total_force = gravity_acc + wind_force

        # --- 4.2 随机化初始状态 ---
        # 随机初始位置
        pos = np.random.randn(3).astype(np.float32) * 0.5
        # 随机初始旋转（四元数形式）
        q = np.random.randn(4).astype(np.float32)
        q /= (np.linalg.norm(q) + 1e-6) # 四元数必须归一化才是合法的旋转
        
        # 将随机的位姿应用到物体上
        actor.set_pose(sapien.Pose(pos, q))
        
        # 随机设置初始线速度
        actor.set_velocity(np.random.randn(3).astype(np.float32) * 3.0)
        
        # === 核心物理：赋予强烈的初始三轴角速度 ===
        # 给定极大的初始角速度（方差为 12.0），使其在空中剧烈翻滚
        w_init = np.random.randn(3).astype(np.float32) * 12.0
        actor.set_angular_velocity(w_init)
        
        traj_x, traj_v = [], []
        
        # --- 4.3 物理预热 (Burn-in) ---
        # 获取当前位姿变换矩阵 (4x4)
        mat = actor.get_pose().to_transformation_matrix()
        # 将局部坐标系的点云转换到世界坐标系。
        # R @ P + T，其中 mat[:3, :3] 是旋转矩阵 R，mat[:3, 3] 是平移向量 T
        pts_prev = (mat[:3, :3] @ canonical_points.T).T + mat[:3, 3]
        
        # 先空转 5 步（使得物理引擎的初始状态更稳定，避免奇异值）
        for _ in range(5):
            # 将环境总力作用在物体的质心 (actor.get_pose().p) 上
            actor.add_force_at_point(total_force, actor.get_pose().p)
            scene.step()

        # --- 4.4 记录轨迹 ---
        for t in range(traj_len):
            # 获取当前时刻的世界坐标系变换矩阵
            mat = actor.get_pose().to_transformation_matrix()
            # 根据当前的位姿，计算出构成该刚体的所有点在世界坐标系下的绝对位置
            pts_curr = (mat[:3, :3] @ canonical_points.T).T + mat[:3, 3]
            
            # 使用欧拉差分计算出每一个点在上一帧到这一帧之间的平均“点速度”
            # 注意：这包含了质心平移速度和刚体旋转带来的线速度
            vel = pts_curr - pts_prev
            
            # 记录当前时刻的位置和速度
            traj_x.append(pts_curr)
            traj_v.append(vel)
            
            # 更新上一时刻的位置，用于下一步计算速度
            pts_prev = pts_curr
            
            # 物理引擎执行 5 次子步仿真 (Sub-stepping)。
            # 这样做可以在保存下来的帧数 (traj_len) 较少的情况下，
            # 让每一帧之间的物理变化更加显著，同时维持内部物理计算的高精度和稳定性。
            for _ in range(5):
                actor.add_force_at_point(total_force, actor.get_pose().p)
                scene.step()
        
        # 将该条轨迹压入总列表中
        all_x.append(np.stack(traj_x)) # 形状: [traj_len, 64, 3]
        all_v.append(np.stack(traj_v)) # 形状: [traj_len, 64, 3]
        
        # 将这 1 个 3D 的重力向量，复制扩展成对应序列长度和通道的张量，方便模型读取
        all_explicit.append(np.tile(gravity_acc.reshape(1, 1, 3), (traj_len, 1, 1)))
        # 同理，复制隐式风力上下文张量
        all_context.append(np.tile(wind_force.reshape(1, 3), (traj_len, 1)))

    # === 5. 将数据打包并保存为 PyTorch 格式 ===
    data_dict = {
        "x": torch.tensor(np.stack(all_x)),                 # [num_traj, traj_len, num_points, 3]
        "v": torch.tensor(np.stack(all_v)),                 # [num_traj, traj_len, num_points, 3]
        "explicit": torch.tensor(np.stack(all_explicit)),   # [num_traj, traj_len, 1, 3]
        "context": torch.tensor(np.stack(all_context))      # [num_traj, traj_len, 3]
    }
    torch.save(data_dict, save_path)

if __name__ == "__main__":
    # 生成 10000 条训练数据
    generate_trajectory_dataset(10000, 60, "data/sapien_train_seq.pt")
    # 生成 200 条验证数据
    generate_trajectory_dataset(200, 60, "data/sapien_val_seq.pt")

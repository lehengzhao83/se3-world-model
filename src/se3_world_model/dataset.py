import os
import torch
from torch.utils.data import Dataset

class SapienSequenceDataset(Dataset):
    """
    Sapien 物理仿真序列数据集类。
    用于从磁盘加载预先生成的轨迹数据，将其切分为固定长度的子序列，
    并对物理状态（位置和速度）进行标准化处理，以便于模型训练。
    """
    def __init__(self, data_path: str, sub_seq_len: int = 6) -> None:
        super().__init__()
        # 检查数据文件路径是否存在，防止由于路径错误导致程序崩溃
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        print(f"Loading sequence dataset: {data_path} ...")
        # 加载保存的 PyTorch 字典数据。
        # 设置 weights_only=True 是一种安全的加载做法，防止加载时执行恶意代码
        self.data = torch.load(data_path, weights_only=True)

        # 提取数据集中的各个物理量和环境信息
        # 假设原始数据维度大致为: [轨迹数量(Num_Traj), 轨迹长度(Traj_Len), 点数(Num_Points), 特征维度]
        self.x = self.data["x"]               # 点云的历史位置数据
        self.v = self.data["v"]               # 点云的历史速度数据
        self.explicit = self.data["explicit"] # 显式的全局向量（如重力等保持等变性的矢量）
        self.context = self.data["context"]   # 隐式的上下文特征（如破坏对称性的环境参数，例如标量风力等）
        
        self.sub_seq_len = sub_seq_len        # 训练时每次采样的子序列长度
        self.num_traj = self.x.shape[0]       # 数据集中包含的独立轨迹（或场景）总数
        self.traj_len = self.x.shape[1]       # 每条轨迹在时间步上的总长度
        
        # 计算全局的均值
        # dim=(0, 1, 2) 表示在"轨迹"、"时间"、"点"这三个维度上求平均，最后仅保留特征维度(3D坐标/速度)
        self.pos_mean = self.x.mean(dim=(0, 1, 2), keepdim=True) 
        self.vel_mean = self.v.mean(dim=(0, 1, 2), keepdim=True)
        
        # === 终极物理修复：统一欧氏度量空间 ===
        # 为什么这样做很重要？
        # 如果对位置(x)和速度(v)使用不同的标准差进行缩放，会破坏三维空间中的欧氏几何度量。
        # 刚体旋转是正交变换，对空间尺度极其敏感。共享同一个缩放因子(shared_std)可以：
        # 1. 强制位置和速度在网络内部维持相同比例的缩放尺度。
        # 2. 保证 SE(3) 等变操作（特别是旋转）在数学上的自洽性。
        
        # 以速度全局标准差为基准（使用 clamp(min=1e-6) 防止数据全部为0时出现除零错误）
        shared_std = self.v.std(dim=(0, 1, 2), keepdim=True).clamp(min=1e-6)
        
        # 强制位置(pos)和速度(vel)共享这同一个标准差进行缩放
        self.pos_std = shared_std
        self.vel_std = shared_std
        
        # 将均值和标准差张量调整为形状 [1, 1, 3] 
        # 这样在 __getitem__ 中做广播运算 (Broadcasting) 时可以正确对齐到具体特征维度
        self.pos_mean = self.pos_mean.view(1, 1, 3)
        self.pos_std = self.pos_std.view(1, 1, 3)
        self.vel_mean = self.vel_mean.view(1, 1, 3)
        self.vel_std = self.vel_std.view(1, 1, 3)

        print(f"Stats loaded. Shared Std: {self.pos_std.mean():.4f}")

    def __len__(self) -> int:
        """
        返回数据集中可供采样的子序列总数。
        """
        # 每条完整的轨迹可以滑动切分出多少个长度为 sub_seq_len 的子序列
        # 例如：轨迹长100，子序列长6，则可切出 100 - 6 = 94 个子序列（下标 0~93）
        samples_per_traj = self.traj_len - self.sub_seq_len
        
        # 保护性检查：如果配置的子序列长度甚至大于原始轨迹长度，说明数据或配置有误
        if samples_per_traj <= 0:
             raise ValueError("Trajectory length too short")
             
        # 数据集总长度 = 轨迹数量 * 每条轨迹可切分的子序列数
        return self.num_traj * samples_per_traj

    def __getitem__(self, idx: int):
        """
        根据全局索引 idx 提取一个子序列及其对应的特征。
        """
        samples_per_traj = self.traj_len - self.sub_seq_len
        
        # 通过整除和取余，将一维的全局索引映射回具体的【轨迹编号】和【起始时间步】
        traj_idx = idx // samples_per_traj     # 确定当前属于哪一条轨迹
        start_t = idx % samples_per_traj       # 确定当前属于这条轨迹的哪个起始时间点
        end_t = start_t + self.sub_seq_len     # 子序列的结束时间点 (不包含 end_t 本身)
        
        # 从该条轨迹中切片截取相应长度的连续时间段
        x_seq = self.x[traj_idx, start_t:end_t]               # [sub_seq_len, N, 3]
        v_seq = self.v[traj_idx, start_t:end_t]               # [sub_seq_len, N, 3]
        explicit_seq = self.explicit[traj_idx, start_t:end_t] # [sub_seq_len, ...]
        context_seq = self.context[traj_idx, start_t:end_t]   # [sub_seq_len, ...]
        
        # 数据标准化：Z-score Normalization (减去均值，除以标准差)
        # 这里的标准差(pos_std和vel_std)在 __init__ 中已被强制统一，保持了物理尺度一致性
        x_norm = (x_seq - self.pos_mean) / self.pos_std
        v_norm = (v_seq - self.vel_mean) / self.vel_std
        
        # 将张量统一转换为 float32 精度返回，供模型前向传播使用
        return x_norm.float(), v_norm.float(), explicit_seq.float(), context_seq.float()

import os

import torch
from torch.utils.data import Dataset


class SapienDataset(Dataset):
    """
    加载离线生成的 SAPIEN 物理数据。
    """
    def __init__(self, data_path: str) -> None:
        super().__init__()
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"找不到数据文件: {data_path}，请先运行 scripts/generate_sapien_data.py")

        print(f"正在加载数据集: {data_path} ...")
        self.data = torch.load(data_path)

        # 提取数据到内存
        self.x_t = self.data["x_t"]
        self.explicit = self.data["explicit"]
        self.context = self.data["context"]
        self.x_next = self.data["x_next"]

        print(f"加载完成，共 {len(self.x_t)} 条样本。")

    def __len__(self) -> int:
        return len(self.x_t)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.x_t[idx],
            self.explicit[idx],
            self.context[idx],
            self.x_next[idx]
        )

## 🛠️ 安装指南 (Installation)

1.  **克隆仓库**
    ```bash
    git clone [https://github.com/lehengzhao83/se3-world-model.git](https://github.com/lehengzhao83/se3-world-model.git)
    cd se3-world-model
    ```

2.  **创建环境** (推荐 Python 3.12)
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/Mac
    # .venv\Scripts\activate   # Windows
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    pip install sapien tqdm matplotlib
    ```

## 🚀 快速开始 (Quick Start)

### 1. 数据生成 (Data Generation)
使用 SAPIEN 物理引擎生成训练和验证数据：
```bash
python scripts/generate_sapien_data.py
生成的数据将保存在 data/ 目录下 (sapien_train.pt, sapien_val.pt)。

2. 模型训练 (Training)
支持单卡及多卡 DDP 训练。

单卡调试:

Bash
python train.py --batch_size 32 --epochs 10
多卡分布式训练 (推荐 8x 4090):

Bash
torchrun --nproc_per_node=8 train.py --batch_size 128 --epochs 50
3. 评估与可视化 (Evaluation)
加载训练好的权重，计算 MSE 指标并生成对比图：

Bash
python evaluate.py
结果图片将保存为 eval_result.png。

📂 项目结构 (Structure)
Plaintext
se3-world-model/
├── .github/              # CI/CD 配置
├── assets/               # 结果展示图片
├── data/                 # 数据集存放目录 (gitignored)
├── checkpoints/          # 模型权重保存目录 (gitignored)
├── scripts/
│   └── generate_sapien_data.py  # SAPIEN 数据生成脚本
├── src/
│   └── se3_world_model/
│       ├── components.py # Encoder/Decoder 组件
│       ├── dataset.py    # 数据加载器
│       ├── forces.py     # 显式/隐式力处理模块
│       ├── layers.py     # Vector Neurons 核心层
│       └── model.py      # 完整的世界模型架构
├── tests/                # 单元测试
├── train.py              # DDP 训练脚本
├── evaluate.py           # 评估与可视化脚本
├── pyproject.toml        # 项目配置 (Linter/Type Checker)
└── requirements.txt      # 依赖列表
🤝 贡献 (Contributing)
本项目执行严格的代码规范。提交代码前请运行以下检查：

Bash
ruff check .
pyright .
python -m unittest discover -s tests

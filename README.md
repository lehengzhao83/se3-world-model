---
# SE(3) World Model 部署与运行指南

本指南基于 **Alibaba Cloud / Ubuntu (dnf/apt 环境)** 且配备 **NVIDIA GPU (如 8x RTX 4090)** 的服务器环境编写。

## 1. 基础系统环境准备

首先需要安装 SAPIEN 引擎和底层渲染所需的系统依赖库：

```bash
# 更新系统元数据并安装必要的系统库
sudo dnf install -y libX11 libXext libXrender libXcomposite libXcursor libXi libXtst mesa-libGL \
                    vulkan-loader mesa-vulkan-drivers llvm-libs libwayland-client nano

```

## 2. 环境配置 (Conda)

建议使用 Miniconda 来管理 Python 环境（Python 3.10 为佳）：

```bash
# 1. 下载并安装 Miniconda (如果尚未安装)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
source ~/.bashrc
$HOME/miniconda/bin/conda init bash
source ~/.bashrc

# 2. 创建并激活 Python 3.10 环境
conda create -n se3 python=3.10 -y
conda activate se3

```

## 3. 安装 Python 依赖

**注意：** 必须安装特定版本的 `sapien` 以匹配脚本 API，并安装支持 CUDA 12.1 的 `torch`。

```bash
# 1. 安装 PyTorch 相关 (CUDA 12.1 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. 安装 SAPIEN 2.2.2 (核心物理仿真引擎)
pip install sapien==2.2.2

# 3. 安装其他工具库
pip install numpy transforms3d tqdm scipy matplotlib

```

## 4. 数据生成 (Data Generation)

使用 SAPIEN 物理引擎生成训练和验证所需的点云序列数据：

```bash
# 设置显卡可见（可选，防止多进程冲突）
export CUDA_VISIBLE_DEVICES=0

# 运行数据生成脚本
python scripts/generate_sapien_data.py

```

*执行成功后，会在 `data/` 目录下生成 `sapien_train.pt` 和 `sapien_val.pt`。*

## 5. 模型训练 (Training)

支持多显卡分布式训练（以 8 张显卡为例）：

```bash
# 确保所有显卡可见并启动分布式训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 --master_port=29500 \
train.py --batch_size 128 --epochs 50

```

## 6. 模型评估与可视化 (Evaluation)

加载训练好的权重进行评估，并生成对比图：

```bash
python evaluate.py

```

*结果将保存为 `eval_result.png`。

```

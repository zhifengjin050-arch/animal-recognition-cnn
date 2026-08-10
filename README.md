# 🐾 Animal Recognition CNN — 基于CNN的常见动物种类识别系统

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/ResNet18-Transfer%20Learning-blue" alt="ResNet18">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Flask-2.0+-000000?style=flat&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/Accuracy-92.29%25-success" alt="Accuracy">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

<p align="center">
  <b>基于 ResNet18 迁移学习的 8 类常见动物智能识别系统</b><br>
  支持 Web 端与桌面 GUI 双端部署 · 10,000 张高质量数据集 · 92.29% 准确率
</p>

---

## 📋 目录

- [项目背景](#项目背景)
- [功能特性](#功能特性)
- [技术架构](#技术架构)
- [模型性能](#模型性能)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
- [数据集说明](#数据集说明)
- [部署指南](#部署指南)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 项目背景

本系统以**8种常见动物**（牛、狗、猫、羊、虎、鸡、鸭、狮子）为识别对象，基于**卷积神经网络（CNN）**和**ResNet18 迁移学习**技术，构建了一套完整的动物种类智能识别解决方案。

### 核心挑战

- 传统图像识别依赖手工特征设计，泛化能力有限
- 动物图像存在姿态多样、背景复杂、光照变化等挑战
- 需要一套端到端的自动化识别系统，兼顾准确率与实时性

### 技术选型

| 技术 | 选型 | 理由 |
|------|------|------|
| 深度学习框架 | PyTorch 2.0+ | 动态计算图，科研社区首选 |
| 骨干网络 | ResNet18 | 轻量高效，适合中小规模数据集 |
| 迁移学习 | ImageNet 预训练 | 加速收敛，提升小样本表现 |
| Web 框架 | Flask | 轻量级 Python Web 框架 |
| 前端 | 原生 HTML5 + Tailwind CSS + Chart.js | 零依赖快速开发 |
| 桌面 GUI | Tkinter | Python 内置跨平台 GUI 库 |

---

## 功能特性

### 🎯 核心功能

- **智能识别**：上传图片即可自动识别 8 种动物，支持 Top-5 多类别概率输出
- **未知拒识**：当置信度低于阈值（60%）时，自动判定为未知类别，避免误判
- **实时推理**：单次推理耗时 < 50ms（GPU），< 200ms（CPU）

### 🖥️ 双端部署

| 部署方式 | 技术栈 | 适用场景 |
|----------|--------|----------|
| **Web 端** | Flask + HTML5 + Chart.js | 远程访问、团队协作、演示展示 |
| **桌面 GUI** | Tkinter + PIL | 本地离线使用、快速体验 |

### 📊 数据可视化

- **雷达图**：各类别识别概率分布
- **柱状图**：各类别概率对比
- **趋势图**：置信度历史变化趋势
- **速度图**：推理耗时统计

### 🔬 数据管理

- **10,000 张数据集**：8 类 × 1,250 张/类，90.4%:4.8%:4.8% 划分
- **三级数据质量保障**：搜索关键词优化 → 文件完整性校验 → 人工抽检复审
- **MD5 去重**：自动检测并报告重复文件
- **数据集快照**：版本化管理，支持差异对比
- **数据溯源**：完整记录数据采集与处理全流程

---

## 技术架构

### 模型架构

- **骨干网络**：ResNet18（预训练 ImageNet 权重）
- **自定义分类头**：
  ```
  Linear(512, 256) → BatchNorm1d(256) → ReLU → Dropout(0.5) → Linear(256, 8)
  ```
- **输入尺寸**：224 × 224 × 3（RGB）
- **输出**：8 类 Softmax 概率分布

### 训练配置

| 参数 | 值 |
|------|------|
| 优化器 | Adam |
| 初始学习率 | 0.001 |
| 学习率调度 | CosineAnnealingLR |
| 批次大小 | 48 |
| 训练轮次 | 15 Epochs |
| 损失函数 | CrossEntropyLoss |
| 数据增强 | RandomResizedCrop + RandomHorizontalFlip + Normalize |

---

## 模型性能

### 整体指标

| 指标 | 数值 |
|------|------|
| **准确率 (Accuracy)** | **92.29%** |
| 宏平均精确率 (Macro Precision) | 91.17% |
| 宏平均召回率 (Macro Recall) | 91.04% |
| 宏平均 F1 分数 (Macro F1-Score) | 91.00% |

### 各类别性能

| 类别 | 精确率 | 召回率 | F1 分数 | 训练样本 |
|------|--------|--------|---------|----------|
| 🐂 牛 | 91.2% | 92.0% | 91.6% | 1,130 |
| 🐕 狗 | 90.5% | 89.8% | 90.1% | 1,130 |
| 🦁 狮子 | 91.8% | 91.5% | 91.6% | 1,130 |
| 🐱 猫 | 92.1% | 91.3% | 91.7% | 1,130 |
| 🐑 羊 | 90.8% | 90.2% | 90.5% | 1,130 |
| 🐯 虎 | 91.5% | 92.3% | 91.9% | 1,130 |
| 🐔 鸡 | 91.0% | 90.8% | 90.9% | 1,130 |
| 🦆 鸭 | 90.5% | 90.4% | 90.4% | 1,130 |

### 超参数对比实验

| 组合 | 学习率 | Batch Size | 最佳准确率 | 验证损失 |
|------|--------|------------|------------|----------|
| **A** ⭐ | **0.001** | **48** | **92.29%** | **0.4645** |
| B | 0.001 | 32 | 91.67% | 0.4780 |
| C | 0.0005 | 48 | 89.17% | 0.5210 |
| D | 0.0005 | 32 | 52.50% | 1.3863 |

---

## 项目结构

```
animal-recognition-cnn/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── src/
│   ├── config.py
│   ├── train_model.py
│   ├── system_train.py
│   ├── app.py
│   ├── gui.py
│   ├── data_manager.py
│   ├── viz_utils.py
│   ├── validate_images.py
│   ├── generate_report.py
│   ├── index.html
│   └── nginx_animal.conf
├── models/
│   └── .gitkeep
├── dataset/
│   └── samples/
└── docs/
    └── images/
```

---

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.6+（GPU 训练推荐，CPU 也可运行）

### 安装

```bash
git clone https://github.com/zhifengjin050-arch/animal-recognition-cnn.git
cd animal-recognition-cnn
pip install -r requirements.txt
```

> 模型权重文件请从 [Releases](https://github.com/zhifengjin050-arch/animal-recognition-cnn/releases) 下载，放入 `models/` 目录。

### 快速体验

**Web 端（推荐）**：
```bash
cd src
python app.py
# 打开浏览器访问 http://127.0.0.1:5000
```

**桌面 GUI**：
```bash
cd src
python gui.py
```

### 模型训练

```bash
cd src
python train_model.py    # 使用预配置数据集训练
python system_train.py   # 或使用爬虫自动采集数据并训练
```

---

## 使用说明

### Web 端功能

1. **动物识别**：上传图片，系统自动识别并返回 Top-5 预测结果
2. **数据可视化**：雷达图、柱状图展示识别概率分布
3. **训练数据管理**：查看数据集详情、训练参数配置
4. **识别历史**：查看历史识别记录，支持清空

### API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/predict` | POST | 上传图片进行识别 |
| `/api/dataset_info` | GET | 获取数据集信息 |
| `/api/model_metrics` | GET | 获取模型性能指标 |
| `/api/history` | GET | 获取识别历史记录 |
| `/api/dashboard` | GET | 获取数据管理仪表盘 |

---

## 数据集说明

### 三级质量保障机制

| 级别 | 方法 | 说明 |
|------|------|------|
| **第一级** | 关键词优化 | 构建复合检索短语，从源头减少噪声 |
| **第二级** | 文件完整性校验 | 基于 PIL 校验图像可解码性 |
| **第三级** | 人工抽检复审 | 人工抽检确认标签正确性 |

### 数据集统计

| 划分 | 样本数 | 占比 |
|------|--------|------|
| 训练集 | 9,040 | 90.4% |
| 验证集 | 480 | 4.8% |
| 测试集 | 480 | 4.8% |
| **总计** | **10,000** | **100%** |

---

## 部署指南

### Nginx + Flask 生产部署

```bash
sudo cp src/nginx_animal.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/nginx_animal.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### Docker 部署

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY src/ ./src/
COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "src/app.py"]
```

---

## 贡献指南

欢迎贡献！请遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范。

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'feat: add amazing feature'`
4. 推送并提交 Pull Request

---

## 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

---

<p align="center">
  <sub>Made with ❤️ by 金智枫 | 广州工商学院 工学院</sub>
</p>
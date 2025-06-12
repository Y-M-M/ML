# CRNN自动化训练测试系统

这是一个完整的自动化系统，可以自动完成数据集划分、模型训练和测试评估的全流程。

## 📁 文件说明

### 核心脚本
- `auto_train_test_pipeline.py` - 主要的自动化流程类
- `run_experiment.py` - 简化的实验运行脚本
- `train_crnn.py` - 原有的训练脚本
- `test_model.py` - 原有的测试脚本

### 数据文件
- `Dataset/` - 原始数据集目录
- `Dataset/labels.csv` - 标签文件
- `yolov8_dataset/` - 自动生成的YOLO格式数据集

## 🚀 快速开始

### 方式1: 一键运行完整流程

```bash
# 运行完整实验（数据划分 + 训练 + 测试）
python run_experiment.py

# 自定义测试集比例为30%
python run_experiment.py --test-size 0.3

# 使用不同的随机种子
python run_experiment.py --random-seed 123
```

### 方式2: 分步骤运行

```bash
# 仅准备数据集
python run_experiment.py --prepare-only

# 仅运行训练
python run_experiment.py --skip-test

# 仅运行测试
python run_experiment.py --skip-train
```

### 方式3: 直接使用主脚本

```python
from auto_train_test_pipeline import AutoTrainTest

# 创建实例
auto_trainer = AutoTrainTest(test_size=0.2, random_state=42)

# 运行完整流程
report = auto_trainer.run_full_pipeline()
```

## 📊 系统功能

### 1. 数据集划分
- ✅ 自动从`Dataset/`目录读取数据
- ✅ 智能划分训练集和验证集
- ✅ 确保同一图像的所有标注在同一集合中
- ✅ 支持自定义划分比例和随机种子

### 2. 目录结构管理
- ✅ 自动创建`yolov8_dataset/`目录结构
- ✅ 复制图像到对应的训练/验证目录
- ✅ 生成YOLO格式的`data.yaml`配置文件

### 3. 自动化训练
- ✅ 调用现有的`train_crnn.py`脚本
- ✅ 捕获训练输出和错误信息
- ✅ 支持训练超时控制（默认1小时）

### 4. 自动化测试
- ✅ 调用现有的`test_model.py`脚本
- ✅ 捕获测试结果和性能指标
- ✅ 支持测试超时控制（默认30分钟）

### 5. 实验报告
- ✅ 自动生成详细的JSON格式实验报告
- ✅ 提取关键性能指标
- ✅ 记录完整的实验配置和时间戳

## 📈 输出结果

### 目录结构
```
project/
├── Dataset/                    # 原始数据
├── yolov8_dataset/            # 自动生成
│   └── images/
│       ├── train/             # 训练集图像
│       └── val/               # 验证集图像
├── results/                   # 实验结果
│   └── experiment_report_*.json
└── data.yaml                  # YOLO配置文件
```

### 实验报告
报告包含以下信息：
- 实验配置（时间戳、数据划分比例、随机种子）
- 训练结果（成功状态、输出、性能指标）
- 测试结果（成功状态、输出、性能指标）
- 错误信息（如果有）

## ⚙️ 配置选项

### 命令行参数
- `--test-size`: 测试集比例（默认0.2，即20%）
- `--random-seed`: 随机种子（默认42）
- `--skip-train`: 跳过训练步骤
- `--skip-test`: 跳过测试步骤
- `--prepare-only`: 仅准备数据集

### 可修改参数
在脚本中可以修改的参数：
- 超时时间（训练1小时，测试30分钟）
- 目录路径
- 脚本文件名

## 🔧 故障排除

### 常见问题

1. **缺少依赖包**
```bash
pip install pandas scikit-learn opencv-python pillow torch torchvision
```

2. **找不到文件**
确保以下文件存在：
- `Dataset/labels.csv`
- `train_crnn.py`
- `test_model.py`

3. **权限问题**
```bash
chmod +x run_experiment.py
```

4. **内存不足**
- 减少批次大小
- 减少训练轮数
- 使用CPU而非GPU

### 调试技巧

1. **仅准备数据集测试**
```bash
python run_experiment.py --prepare-only
```

2. **查看详细输出**
实验报告会保存在`results/`目录中，包含完整的训练和测试输出。

3. **分步执行**
```bash
# 先准备数据
python run_experiment.py --prepare-only

# 再执行训练
python run_experiment.py --skip-test

# 最后执行测试
python run_experiment.py --skip-train
```

## 📝 示例用法

### 基础使用
```bash
# 默认配置运行
python run_experiment.py

# 查看帮助
python run_experiment.py --help
```

### 高级配置
```bash
# 使用30%作为测试集，随机种子为123
python run_experiment.py --test-size 0.3 --random-seed 123

# 仅准备数据集和训练
python run_experiment.py --skip-test

# 快速验证数据集划分
python run_experiment.py --prepare-only
```

### Python代码使用
```python
from auto_train_test_pipeline import AutoTrainTest

# 创建实例并运行
trainer = AutoTrainTest(test_size=0.25, random_state=100)
report = trainer.run_full_pipeline()

# 查看结果
print(f"训练成功: {report['training']['success']}")
print(f"测试成功: {report['testing']['success']}")
```

## 💡 提示

1. **首次运行**建议使用`--prepare-only`先检查数据集划分
2. **长时间训练**可以使用`nohup`在后台运行
3. **多次实验**修改随机种子可以得到不同的数据划分
4. **调试模式**可以先跳过耗时的训练或测试步骤

---

📧 如有问题，请检查实验报告中的详细错误信息。 


# 验收测试
process_labels.py  修改labels.csv文件的路径，运行即可输出数据到results.csv文件

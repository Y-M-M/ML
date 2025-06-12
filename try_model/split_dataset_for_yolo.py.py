import os
import shutil
import random

# 配置
DATASET_DIR = './Dataset'  # 原始图像 + labels.csv 所在目录
YOLO_DATASET_DIR = './yolov8_dataset'
SPLIT_RATIO = 0.8  # 训练集比例

# 清理并重新创建YOLO格式目录
folders = [
    f'{YOLO_DATASET_DIR}/images/train',
    f'{YOLO_DATASET_DIR}/images/val',
    f'{YOLO_DATASET_DIR}/labels/train',
    f'{YOLO_DATASET_DIR}/labels/val'
]

# 清理旧文件
for folder in folders:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

# 收集图片文件列表
image_files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.png'))]
print(f"总图像数量: {len(image_files)}")

# 设置随机种子以确保可重复性
random.seed(42)
random.shuffle(image_files)

# 划分
split_idx = int(len(image_files) * SPLIT_RATIO)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

print(f"训练集: {len(train_files)} 张")
print(f"验证集: {len(val_files)} 张")

# 拷贝图像和标签
for phase, files in [('train', train_files), ('val', val_files)]:
    copied_images = 0
    copied_labels = 0
    
    for img_file in files:
        # 复制图像文件
        src_img_path = os.path.join(DATASET_DIR, img_file)
        dst_img_path = os.path.join(YOLO_DATASET_DIR, f'images/{phase}/{img_file}')
        shutil.copyfile(src_img_path, dst_img_path)
        copied_images += 1

        # 复制YOLO标签文件
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        src_label_path = os.path.join(YOLO_DATASET_DIR, 'labels', label_file)
        dst_label_path = os.path.join(YOLO_DATASET_DIR, f'labels/{phase}/{label_file}')
        if os.path.exists(src_label_path):
            shutil.copyfile(src_label_path, dst_label_path)
            copied_labels += 1
        else:
            print(f"⚠️ 缺少标签文件: {label_file}")
    
    print(f"{phase}: 复制了 {copied_images} 张图像，{copied_labels} 个标签")

print("✅ 数据集已成功划分为训练集和验证集！")

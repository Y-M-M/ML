#!/usr/bin/env python3
"""
CRNN改进训练脚本 - 方案B
基于79.76%有效架构，进行最小化、经过验证的改进
目标：稳步提升到85%+
"""

import os
import csv
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import random
from tqdm import tqdm
import pandas as pd

# 配置参数
TRAIN_IMAGES_DIR = './yolov8_dataset/images/train'
VAL_IMAGES_DIR = './yolov8_dataset/images/val'
LABEL_FILE = './Dataset/labels.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 🎯 保持与有效模型完全一致的核心参数
IMG_HEIGHT = 64
IMG_WIDTH = 256
HIDDEN_SIZE = 256  # 保持256，不改变
NUM_LAYERS = 2
CHARS = '0123456789.'
BLANK_IDX = len(CHARS)
char2idx = {c: i for i, c in enumerate(CHARS)}

# 训练参数 - 保守改进
BATCH_SIZE = 16
EPOCHS = 60        # 适度增加训练轮数
LEARNING_RATE = 0.0008  # 稍微降低学习率
EARLY_STOP_PATIENCE = 15

class ImprovedCRNN(nn.Module):
    """改进的CRNN - 基于有效架构，最小化改动"""
    
    def __init__(self, img_h, n_channels, n_hidden, n_classes):
        super().__init__()
        
        # 🎯 保持与有效模型完全一致的CNN架构
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU()  # 🎯 保持关键的最后一层
        )
        
        # 🎯 保持与有效模型完全一致的RNN架构
        self.rnn = nn.LSTM(512, n_hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        
        # 🎯 保持简单的输出层
        self.fc = nn.Linear(n_hidden * 2, n_classes + 1)

    def forward(self, x):
        # 🎯 保持与有效模型完全一致的前向传播
        conv = self.cnn(x)
        conv = F.adaptive_avg_pool2d(conv, (1, conv.size(3)))
        seq = conv.squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(seq)
        logits = self.fc(rnn_out)
        return logits.permute(1, 0, 2)

class ImprovedDataset(Dataset):
    """改进的数据集 - 基于有效的数据处理，添加轻度增强"""
    
    def __init__(self, data, chars, img_h, img_w, training=True):
        self.data = data
        self.chars = chars
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.img_h = img_h
        self.img_w = img_w
        self.training = training
        
        # 🎯 保持与有效模型一致的基础变换
        self.transform = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __len__(self):
        return len(self.data)
    
    def apply_light_augmentation(self, image):
        """轻度数据增强 - 只在训练时使用"""
        if not self.training or random.random() > 0.25:  # 只有25%概率增强
            return image
        
        # 非常保守的增强
        if random.random() < 0.5:
            # 轻微对比度调整
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.95, 1.05)  # 很小的变化
            image = enhancer.enhance(factor)
        
        return image
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # 加载图像
        if self.training:
            img_path = os.path.join(TRAIN_IMAGES_DIR, item['filename'])
        else:
            img_path = os.path.join(VAL_IMAGES_DIR, item['filename'])
        
        image = cv2.imread(img_path)
        if image is None:
            image = np.ones((50, 100, 3), dtype=np.uint8) * 255
        
        # 使用正确的列名
        x1, y1, x2, y2 = int(item['xmin']), int(item['ymin']), int(item['xmax']), int(item['ymax'])
        digit_region = image[y1:y2, x1:x2]
        
        if digit_region.size == 0:
            digit_region = np.ones((50, 100, 3), dtype=np.uint8) * 255
        
        # 转换为灰度
        digit_gray = cv2.cvtColor(digit_region, cv2.COLOR_BGR2GRAY)
        pil_image = Image.fromarray(digit_gray)
        
        # 应用轻度数据增强
        pil_image = self.apply_light_augmentation(pil_image)
        
        # 转换为tensor
        image_tensor = self.transform(pil_image)
        
        # 🎯 保持与有效模型一致的标签处理
        label = str(item['number'])
        # 移除前导零，但保留单个'0'
        label = label.lstrip('0') or '0'
        label_indices = [self.char_to_idx[char] for char in label if char in self.char_to_idx]
        
        return image_tensor, torch.tensor(label_indices, dtype=torch.long)

def collate_fn(batch):
    """🎯 保持与有效模型完全一致的collate函数"""
    images, labels = zip(*batch)
    images = torch.stack(images)
    
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    targets = torch.cat(labels)
    
    batch_size = images.size(0)
    # 🎯 关键：使用与有效模型一致的序列长度计算
    # 根据CNN架构计算实际输出长度
    input_lengths = torch.full((batch_size,), 63, dtype=torch.long)  # 经验值，与有效模型一致
    
    return images, targets, input_lengths, label_lengths

def load_and_prepare_data():
    """加载和准备数据"""
    print("📂 加载数据...")
    
    df = pd.read_csv(LABEL_FILE)
    
    train_images = set(os.listdir(TRAIN_IMAGES_DIR))
    val_images = set(os.listdir(VAL_IMAGES_DIR))
    
    train_data = df[df['filename'].isin(train_images)].reset_index(drop=True)
    val_data = df[df['filename'].isin(val_images)].reset_index(drop=True)
    
    print(f"🎯 训练集: {len(train_data)}个样本")
    print(f"🎯 验证集: {len(val_data)}个样本")
    
    return train_data, val_data

def train_improved_crnn():
    """改进的CRNN训练"""
    print("🚀 改进CRNN训练 - 方案B")
    print("=" * 60)
    print(f"🎯 基于79.76%有效架构，最小化改进")
    print(f"📊 训练配置：")
    print(f"   - 训练轮数: {EPOCHS}")
    print(f"   - 批次大小: {BATCH_SIZE}")
    print(f"   - 学习率: {LEARNING_RATE}")
    print(f"   - 隐藏层大小: {HIDDEN_SIZE} (保持不变)")
    print(f"   - 早停耐心值: {EARLY_STOP_PATIENCE}")
    
    # 加载数据
    train_data, val_data = load_and_prepare_data()
    
    # 创建数据集
    train_dataset = ImprovedDataset(train_data, CHARS, IMG_HEIGHT, IMG_WIDTH, training=True)
    val_dataset = ImprovedDataset(val_data, CHARS, IMG_HEIGHT, IMG_WIDTH, training=False)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    # 创建模型
    model = ImprovedCRNN(IMG_HEIGHT, 1, HIDDEN_SIZE, len(CHARS)).to(DEVICE)
    
    model_size = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"📏 模型大小: {model_size:.1f}M 参数")
    
    # 🎯 保持与有效模型相似的优化器配置
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)  # 轻度正则化
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=8, verbose=True)
    
    # 🎯 保持与有效模型一致的损失函数
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    
    # 训练状态
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = []
    
    print(f"\n🔥 开始改进训练...")
    
    for epoch in range(EPOCHS):
        print(f"\n📅 Epoch {epoch+1}/{EPOCHS}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Training')
        for images, targets, input_lengths, target_lengths in progress_bar:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            input_lengths = input_lengths.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # 🎯 保持与有效模型一致的损失计算
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 梯度裁剪
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else float('inf')
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        print("🔍 验证中...")
        with torch.no_grad():
            for images, targets, input_lengths, target_lengths in tqdm(val_loader, desc='Validation'):
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)
                input_lengths = input_lengths.to(DEVICE)
                target_lengths = target_lengths.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, targets, input_lengths, target_lengths)
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录训练历史
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr
        })
        
        print(f"📊 训练损失: {avg_train_loss:.4f}")
        print(f"📊 验证损失: {avg_val_loss:.4f}")
        print(f"📊 学习率: {current_lr:.6f}")
        
        # 早停和模型保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), './crnn_improved_best.pth')
            print(f"💾 保存最佳模型 (验证损失: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"⏳ 早停计数: {patience_counter}/{EARLY_STOP_PATIENCE}")
            
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"⏹️ 早停于第 {epoch+1} 轮")
                break
    
    # 保存最终模型
    final_model_path = './crnn_improved_final.pth'
    torch.save(model.state_dict(), final_model_path)
    
    print(f"\n✅ 改进CRNN训练完成！")
    print(f"📊 总训练轮数: {len(training_history)}")
    print(f"📈 最佳验证损失: {best_val_loss:.4f}")
    print(f"💾 最佳模型: ./crnn_improved_best.pth")
    print(f"💾 最终模型: {final_model_path}")
    
    # 保存训练历史
    history_file = 'training_history_improved.csv'
    pd.DataFrame(training_history).to_csv(history_file, index=False)
    print(f"📊 训练历史已保存到: {history_file}")

if __name__ == '__main__':
    train_improved_crnn() 
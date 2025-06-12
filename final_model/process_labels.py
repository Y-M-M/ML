#!/usr/bin/env python3
"""
从labels.csv文件中读取电表图像的边界框坐标
使用CRNN模型识别电表读数
输出格式: id,reading
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import re

# 配置参数
LABELS_FILE = '/Volumes/YMM/Dataset/labels.csv'  # 标签文件路径
IMAGE_DIR = '/Volumes/YMM/Dataset'  # 图片文件夹路径
CRNN_MODEL_PATH = './crnn_improved_best.pth'
OUTPUT_FILE = './results.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型参数 - 与训练时保持一致
IMG_HEIGHT = 64
IMG_WIDTH = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 2
CHARS = '0123456789.'
BLANK_IDX = len(CHARS)
idx2char = {i: c for i, c in enumerate(CHARS)}

class ImprovedCRNN(nn.Module):
    """改进的CRNN - 与训练时完全一致"""
    
    def __init__(self, img_h, n_channels, n_hidden, n_classes):
        super().__init__()
        
        # CNN架构
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU()
        )
        
        # RNN架构
        self.rnn = nn.LSTM(512, n_hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        
        # 输出层
        self.fc = nn.Linear(n_hidden * 2, n_classes + 1)

    def forward(self, x):
        conv = self.cnn(x)
        conv = F.adaptive_avg_pool2d(conv, (1, conv.size(3)))
        seq = conv.squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(seq)
        logits = self.fc(rnn_out)
        return logits.permute(1, 0, 2)

# CTC解码

def decode_ctc_greedy(output, blank_idx=BLANK_IDX):
    predictions = torch.argmax(output, dim=2).squeeze(1).cpu().numpy()
    
    result = []
    prev_idx = -1
    
    for idx in predictions:
        if idx != blank_idx and idx != prev_idx:
            if idx < len(idx2char):
                result.append(idx2char[idx])
        prev_idx = idx
    
    return ''.join(result)

# 智能后处理器

class SmartPostProcessor:
    """智能后处理器 - 清理和格式化电表读数"""
    
    def __init__(self):
        # 电表读数的常见模式
        self.common_patterns = [
            r'^\d{1,6}\.\d{1,2}$',  # 标准格式: 1234.5
            r'^\d{1,6}$',           # 整数格式: 1234
        ]
    
    def smart_clean(self, text):
        """智能清理和格式化"""
        if not text:
            return ""
        
        # 基础清理 - 只保留数字和小数点
        cleaned = ''.join(c for c in text if c.isdigit() or c == '.')
        
        if not cleaned:
            return ""
        
        # 处理多个连续小数点
        cleaned = re.sub(r'\.{2,}', '.', cleaned)
        
        # 处理尾部小数点
        if cleaned.endswith('.'):
            cleaned = cleaned.rstrip('.')
        
        # 处理开头小数点
        if cleaned.startswith('.'):
            cleaned = '0' + cleaned
        
        # 处理多个小数点
        if cleaned.count('.') > 1:
            parts = cleaned.split('.')
            if len(parts) > 2:
                cleaned = parts[0] + '.' + ''.join(parts[1:])
        
        # 处理前导零
        if cleaned and not cleaned.startswith('.'):
            if '.' in cleaned:
                integer_part, decimal_part = cleaned.split('.', 1)
                integer_part = integer_part.lstrip('0') or '0'
                cleaned = f"{integer_part}.{decimal_part}"
            else:
                cleaned = cleaned.lstrip('0') or '0'
        
        # 验证格式合理性
        if not self.is_reasonable_meter_reading(cleaned):
            cleaned = self.try_fix_format(cleaned)
        
        return cleaned
    
    def is_reasonable_meter_reading(self, text):
        """检查是否是合理的电表读数格式"""
        if not text:
            return False
        
        # 基本格式检查
        for pattern in self.common_patterns:
            if re.match(pattern, text):
                return True
        
        # 长度检查 (电表读数通常不会太长)
        if len(text) > 10:
            return False
        
        # 小数位检查 (通常不超过2位小数)
        if '.' in text:
            decimal_part = text.split('.')[1]
            if len(decimal_part) > 2:
                return False
        
        return True
    
    def try_fix_format(self, text):
        """尝试修复异常格式"""
        if not text:
            return text
        
        # 如果小数部分太长，截断到2位
        if '.' in text:
            integer_part, decimal_part = text.split('.', 1)
            if len(decimal_part) > 2:
                decimal_part = decimal_part[:2]
                text = f"{integer_part}.{decimal_part}"
        
        return text
    
    def process_prediction(self, raw_prediction):
        """处理单个预测结果"""
        if not raw_prediction:
            return ""
        
        # 智能清理
        cleaned = self.smart_clean(raw_prediction)
        
        return cleaned

# 处理数据集

def process_labels():
    """从labels.csv文件中读取边界框并识别电表读数"""
    print("🚀 开始处理数据集...")
    print("=" * 60)
    
    # 检查标签文件
    if not os.path.exists(LABELS_FILE):
        print(f"⚠️  标签文件不存在: {LABELS_FILE}")
        return
    
    # 加载CRNN模型
    try:
        crnn_model = ImprovedCRNN(IMG_HEIGHT, 1, HIDDEN_SIZE, len(CHARS)).to(DEVICE)
        crnn_model.load_state_dict(torch.load(CRNN_MODEL_PATH, map_location=DEVICE))
        crnn_model.eval()
        print("✅ CRNN模型加载成功")
    except Exception as e:
        print(f"❌ CRNN模型加载失败: {e}")
        return
    
    # 读取标签文件
    try:
        labels_df = pd.read_csv(LABELS_FILE)
        print("✅ 标签文件加载成功")
    except Exception as e:
        print(f"❌ 标签文件加载失败: {e}")
        return
    
    # 图像预处理transform
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 初始化后处理器
    postprocessor = SmartPostProcessor()
    
    # 存储结果
    results = []
    
    print(f"\n🔍 开始识别电表读数...")
    
    for i, row in labels_df.iterrows():
        filename = row['filename']
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        
        try:
            # 构建完整的图片路径
            if os.path.isabs(filename):
                # 如果filename已经是绝对路径，直接使用
                image_path = filename
            else:
                # 如果是相对路径或只是文件名，与IMAGE_DIR组合
                image_path = os.path.join(IMAGE_DIR, filename)
            
            # 加载图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️  无法读取图像: {image_path}")
                continue
            
            # 提取数字区域
            digit_region = image[ymin:ymax, xmin:xmax]
            if digit_region.size == 0:
                print(f"⚠️  无法提取有效数字区域: {filename}")
                continue
            
            # 预处理数字区域
            digit_gray = cv2.cvtColor(digit_region, cv2.COLOR_BGR2GRAY)
            pil_image = Image.fromarray(digit_gray)
            image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
            
            # CRNN识别
            with torch.no_grad():
                outputs = crnn_model(image_tensor)
                raw_prediction = decode_ctc_greedy(outputs)
            
            # 智能后处理
            processed_prediction = postprocessor.process_prediction(raw_prediction)
            
            # 记录结果
            results.append({
                'id': str(i + 1),
                'reading': processed_prediction
            })
            
            # 显示进度
            if (i + 1) % 10 == 0 or i == len(labels_df) - 1:
                print(f"进度: {i + 1}/{len(labels_df)} - {filename}: {processed_prediction}")
        
        except Exception as e:
            print(f"❌ 处理 {filename} 时出错: {e}")
            continue
    
    # 保存结果到CSV
    if results:
        df = pd.DataFrame(results)
        
        # 保存为CSV格式，不包含索引和表头
        df.to_csv(OUTPUT_FILE, index=False, header=False)
        
        print(f"\n" + "=" * 60)
        print(f"✅ 处理完成！")
        print(f"📊 成功处理: {len(results)} 个图像")
        print(f"💾 结果已保存到: {OUTPUT_FILE}")
        print(f"📝 格式: id,reading (如: 1,8430.8)")
        
        # 显示前几行结果作为示例
        print(f"\n📋 前5行结果示例:")
        for i, row in df.head().iterrows():
            print(f"   {row['id']},{row['reading']}")
        
    else:
        print(f"❌ 没有成功处理任何图像")

if __name__ == '__main__':
    process_labels() 
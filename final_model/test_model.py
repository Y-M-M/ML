#!/usr/bin/env python3
"""
测试改进CRNN模型
基于智能后处理推理，使用新训练的改进模型
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
from ultralytics import YOLO
import re

# 配置参数
VAL_IMAGES_DIR = './yolov8_dataset/images/val'
LABEL_FILE = './Dataset/labels.csv'
YOLO_MODEL_PATH = './runs/detect/meter_detection/weights/best.pt'
CRNN_MODEL_PATH = './crnn_improved_best.pth'  # 使用改进的最佳模型
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 🎯 保持与有效模型完全一致的参数
IMG_HEIGHT = 64
IMG_WIDTH = 256
HIDDEN_SIZE = 256  # 保持256
NUM_LAYERS = 2
CHARS = '0123456789.'
BLANK_IDX = len(CHARS)
idx2char = {i: c for i, c in enumerate(CHARS)}

class ImprovedCRNN(nn.Module):
    """改进的CRNN - 与训练时完全一致"""
    
    def __init__(self, img_h, n_channels, n_hidden, n_classes):
        super().__init__()
        
        # 🎯 保持与有效模型完全一致的CNN架构
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU()
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

def decode_ctc_greedy(output, blank_idx=BLANK_IDX):
    """🎯 保持与智能后处理完全一致的CTC解码"""
    predictions = torch.argmax(output, dim=2).squeeze(1).cpu().numpy()
    
    result = []
    prev_idx = -1
    
    for idx in predictions:
        if idx != blank_idx and idx != prev_idx:
            if idx < len(idx2char):
                result.append(idx2char[idx])
        prev_idx = idx
    
    return ''.join(result)

class SmartPostProcessor:
    """🎯 与智能后处理完全一致的后处理器"""
    
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
        
        # 步骤1: 基础清理 - 只保留数字和小数点
        cleaned = ''.join(c for c in text if c.isdigit() or c == '.')
        
        if not cleaned:
            return ""
        
        # 步骤2: 处理多个连续小数点
        cleaned = re.sub(r'\.{2,}', '.', cleaned)  # 多个点变成一个
        
        # 步骤3: 处理尾部小数点 - 关键修复！
        if cleaned.endswith('.'):
            # 如果以点结尾，删除尾部点
            cleaned = cleaned.rstrip('.')
        
        # 步骤4: 处理开头小数点
        if cleaned.startswith('.'):
            cleaned = '0' + cleaned
        
        # 步骤5: 处理多个小数点
        if cleaned.count('.') > 1:
            # 保留第一个小数点，其余变成数字
            parts = cleaned.split('.')
            if len(parts) > 2:
                # 例如: "12.34.56" -> "12.3456"
                cleaned = parts[0] + '.' + ''.join(parts[1:])
        
        # 步骤6: 处理前导零
        if cleaned and not cleaned.startswith('.'):
            if '.' in cleaned:
                integer_part, decimal_part = cleaned.split('.', 1)
                integer_part = integer_part.lstrip('0') or '0'
                # 保留所有小数位，不删除尾部0
                cleaned = f"{integer_part}.{decimal_part}"
            else:
                cleaned = cleaned.lstrip('0') or '0'
        
        # 步骤7: 验证格式合理性
        if not self.is_reasonable_meter_reading(cleaned):
            # 如果格式不合理，尝试修复
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
        if len(text) > 10:  # 超过10位可能有问题
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

def smart_bbox_fallback(yolo_results, true_bbox, img_w=400, img_h=296):
    """🎯 与智能后处理完全一致的边界框选择"""
    if len(yolo_results[0].boxes) > 0:
        box = yolo_results[0].boxes[0]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        width = x2 - x1
        height = y2 - y1
        
        if confidence < 0.3 or width < 20 or height < 10 or width > 300 or height > 100:
            x1, y1, x2, y2 = true_bbox
            source = "真实框(YOLO低质量)"
        else:
            source = f"YOLO(置信度{confidence:.2f})"
    else:
        x1, y1, x2, y2 = true_bbox
        source = "真实框(YOLO无检测)"
    
    return int(x1), int(y1), int(x2), int(y2), source

def test_improved_model():
    """测试改进CRNN模型"""
    print("🚀 改进CRNN模型测试 - 方案B")
    print("基于79.76%智能后处理，使用改进的CRNN模型")
    print("=" * 60)
    
    # 检查模型文件
    if not os.path.exists(CRNN_MODEL_PATH):
        print(f"❌ 改进模型不存在: {CRNN_MODEL_PATH}")
        print("💡 请先运行 train_crnn_improved.py 训练模型")
        return
    
    # 加载YOLO模型
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO模型加载成功")
    except Exception as e:
        print(f"❌ YOLO模型加载失败: {e}")
        return
    
    # 加载改进的CRNN模型
    try:
        crnn_model = ImprovedCRNN(IMG_HEIGHT, 1, HIDDEN_SIZE, len(CHARS)).to(DEVICE)
        crnn_model.load_state_dict(torch.load(CRNN_MODEL_PATH, map_location=DEVICE))
        crnn_model.eval()
        print("✅ 改进CRNN模型加载成功")
        
        # 检查模型大小
        model_size = sum(p.numel() for p in crnn_model.parameters()) / 1e6
        print(f"📏 模型大小: {model_size:.1f}M 参数")
    except Exception as e:
        print(f"❌ CRNN模型加载失败: {e}")
        return
    
    # 加载验证数据
    df = pd.read_csv(LABEL_FILE)
    val_images = set(os.listdir(VAL_IMAGES_DIR))
    val_data = df[df['filename'].isin(val_images)].reset_index(drop=True)
    print(f"📊 验证集样本数: {len(val_data)}")
    
    # 🎯 保持与智能后处理完全一致的图像预处理
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 初始化智能后处理器
    postprocessor = SmartPostProcessor()
    
    print(f"\n🔍 开始改进模型推理...")
    
    correct_predictions = 0
    total_predictions = 0
    yolo_used = 0
    fallback_used = 0
    postprocess_fixes = 0
    
    predictions_data = []
    
    for idx, row in val_data.iterrows():
        filename = row['filename']
        true_number = str(row['number'])
        img_path = os.path.join(VAL_IMAGES_DIR, filename)
        
        # 加载图像
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # YOLO检测
        results = yolo_model(image, verbose=False)
        
        # 智能边界框选择
        true_bbox = (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))
        x1, y1, x2, y2, bbox_source = smart_bbox_fallback(results, true_bbox)
        
        if "YOLO" in bbox_source:
            yolo_used += 1
        else:
            fallback_used += 1
        
        # 提取数字区域
        digit_region = image[y1:y2, x1:x2]
        if digit_region.size == 0:
            continue
        
        # 预处理数字区域
        digit_gray = cv2.cvtColor(digit_region, cv2.COLOR_BGR2GRAY)
        pil_image = Image.fromarray(digit_gray)
        image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
        
        # CRNN推理
        with torch.no_grad():
            outputs = crnn_model(image_tensor)
            raw_prediction = decode_ctc_greedy(outputs)
        
        # 智能后处理
        original_prediction = raw_prediction
        processed_prediction = postprocessor.process_prediction(raw_prediction)
        
        # 检查是否被后处理修复
        postprocess_info = ""
        if original_prediction != processed_prediction:
            postprocess_fixes += 1
            postprocess_info = f" [原始='{original_prediction}']"
        
        # 判断准确性
        is_correct = processed_prediction == true_number
        if is_correct:
            correct_predictions += 1
        
        total_predictions += 1
        
        # 记录预测结果
        predictions_data.append({
            'filename': filename,
            'predicted': processed_prediction,
            'actual': true_number,
            'correct': is_correct,
            'bbox_source': bbox_source,
            'raw_prediction': original_prediction
        })
        
        # 显示结果（只显示前40个和错误案例）
        status = "✅" if is_correct else "❌"
        if idx < 40 or not is_correct:
            print(f"{filename}: 预测='{processed_prediction}' | 真实='{true_number}' | {bbox_source}{postprocess_info} | {status}")
    
    # 计算准确率
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    print(f"\n" + "=" * 60)
    print(f"📊 改进CRNN模型准确率: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
    print(f"📍 边界框使用: YOLO={yolo_used}, 回退={fallback_used}")
    print(f"🔧 后处理修复: {postprocess_fixes}个案例")
    
    # 与基线比较
    baseline_accuracy = 79.76
    improvement = accuracy - baseline_accuracy
    if improvement > 0:
        print(f"📈 相比基线提升: +{improvement:.2f}%")
        print(f"✨ 改进成功！")
        if accuracy >= 85:
            print(f"🎉 达到85%+目标！")
    elif improvement < -1:
        print(f"📉 相比基线下降: {improvement:.2f}%")
        print(f"⚠️ 改进未达预期")
    else:
        print(f"📊 与基线基本持平: {improvement:+.2f}%")
    
    print(f"🎯 距离90%目标还差: {90 - accuracy:.2f}%")
    
    # 保存结果
    results_file = 'validation_predictions_improved.csv'
    pd.DataFrame(predictions_data).to_csv(results_file, index=False)
    print(f"💾 保存结果到: {results_file}")
    print(f"✅ 改进CRNN模型测试完成")

if __name__ == '__main__':
    test_improved_model() 
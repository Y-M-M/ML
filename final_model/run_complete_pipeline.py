#!/usr/bin/env python3
"""
电表读数识别项目 - 完整训练和推理流程
直接使用 YOLO 命令行和 data.yaml 进行训练，无需 train_yolo.py
"""
import os
import subprocess
import sys
import argparse

def run_command(cmd, description=""):
    """运行命令并处理输出"""
    print(f"\n🚀 {description}")
    print(f"执行命令: {cmd}")
    print("-" * 60)
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ 命令执行失败: {cmd}")
        sys.exit(1)
    else:
        print(f"✅ {description} 完成")
    
    return result

def check_data_ready():
    """检查数据是否准备就绪"""
    required_files = [
        './Dataset/labels.csv',
        './data.yaml',
        './yolov8_dataset/images/train',
        './yolov8_dataset/images/val',
        './yolov8_dataset/labels/train',
        './yolov8_dataset/labels/val'
    ]
    
    print("🔍 检查数据准备状态...")
    all_ready = True
    
    for path in required_files:
        if os.path.exists(path):
            if os.path.isdir(path):
                count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.txt'))])
                print(f"✅ {path}: {count} 个文件")
            else:
                print(f"✅ {path}: 存在")
        else:
            print(f"❌ {path}: 不存在")
            all_ready = False
    
    return all_ready

def main():
    parser = argparse.ArgumentParser(description='电表读数识别完整流程')
    parser.add_argument('--step', choices=['data', 'yolo', 'crnn', 'inference', 'all'], 
                        default='all', help='执行的步骤')
    parser.add_argument('--inference_mode', choices=['validation', 'all'], 
                        default='validation', help='推理模式')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎯 电表读数识别项目 - 完整训练流程")
    print("=" * 80)
    
    if args.step in ['data', 'all']:
        print("\n📊 第1步: 数据准备")
        
        if not check_data_ready():
            print("\n🔧 开始数据准备...")
            
            # 转换标签格式
            run_command("python convert_labels_for_yolo.py", 
                       "转换标签为YOLO格式")
            
            # 划分数据集
            run_command("python split_dataset_for_yolo.py.py", 
                       "划分训练集和验证集")
            
            print("✅ 数据准备完成!")
        else:
            print("✅ 数据已准备就绪，跳过数据准备步骤")
    
    if args.step in ['yolo', 'all']:
        print("\n🎯 第2步: YOLOv8检测模型训练")
        
        if os.path.exists('runs/detect/meter_detection/weights/best.pt'):
            print("✅ YOLOv8模型已存在，跳过训练")
        else:
            # 检查必要文件是否存在
            if not os.path.exists('data.yaml'):
                print("❌ data.yaml 文件不存在")
                sys.exit(1)
            if not os.path.exists('yolo11n.pt'):
                print("❌ yolo11n.pt 预训练模型不存在")
                sys.exit(1)
                
            run_command("yolo detect train data=data.yaml model=yolo11n.pt epochs=100 imgsz=640 batch=16 project=runs/detect name=meter_detection patience=20 save=True plots=True val=True save_period=10", 
                       "使用data.yaml训练YOLOv8检测模型")
    
    if args.step in ['crnn', 'all']:
        print("\n🔤 第3步: CRNN识别模型训练")
        
        if os.path.exists('crnn_recognizer_best.pth'):
            print("✅ CRNN模型已存在，跳过训练")
        else:
            run_command("python train_crnn.py", 
                       "训练CRNN识别模型")
    
    if args.step in ['inference', 'all']:
        print("\n🔍 第4步: 端到端推理测试")
        
        # 检查模型文件是否存在
        required_models = [
            'runs/detect/meter_detection/weights/best.pt',
            'crnn_recognizer_best.pth'
        ]
        
        missing_models = [m for m in required_models if not os.path.exists(m)]
        if missing_models:
            print("❌ 缺少模型文件:")
            for model in missing_models:
                print(f"   - {model}")
            print("请先训练相应的模型!")
            sys.exit(1)
        
        run_command(f"python inference.py --mode {args.inference_mode}", 
                   f"在{args.inference_mode}模式下进行推理")
    
    print("\n" + "=" * 80)
    print("🎉 电表读数识别项目流程执行完成!")
    print("=" * 80)
    
    # 输出结果文件
    print("\n📁 生成的文件:")
    output_files = [
        'runs/detect/meter_detection/weights/best.pt',
        'crnn_recognizer_best.pth',
        f'{args.inference_mode}_predictions.csv' if args.step in ['inference', 'all'] else None
    ]
    
    for file in output_files:
        if file and os.path.exists(file):
            size = os.path.getsize(file) / (1024*1024)  # MB
            print(f"✅ {file} ({size:.1f}MB)")

if __name__ == '__main__':
    main() 
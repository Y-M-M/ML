#!/usr/bin/env python3
"""
自动化训练测试脚本
功能：
1. 自动划分数据集为训练集和测试集
2. 自动运行训练流程
3. 自动运行测试评估
4. 生成完整的实验报告
"""

import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import subprocess
import sys
import json
from datetime import datetime
import cv2
from pathlib import Path

class AutoTrainTest:
    def __init__(self, test_size=0.2, random_state=42):
        """
        初始化自动训练测试系统
        
        Args:
            test_size: 测试集比例 (默认20%)
            random_state: 随机种子 (默认42)
        """
        self.test_size = test_size
        self.random_state = random_state
        
        # 目录配置
        self.dataset_dir = './Dataset'
        self.labels_file = os.path.join(self.dataset_dir, 'labels.csv')
        self.yolo_dataset_dir = './yolov8_dataset'
        self.train_images_dir = os.path.join(self.yolo_dataset_dir, 'images', 'train')
        self.val_images_dir = os.path.join(self.yolo_dataset_dir, 'images', 'val')
        
        # 脚本路径
        self.train_script = 'train_crnn.py'
        self.test_script = 'test_model.py'
        
        # 结果文件
        self.results_dir = './results'
        self.report_file = os.path.join(self.results_dir, f'experiment_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        print("🚀 自动化训练测试系统初始化完成")
        print(f"📊 数据集划分：训练集 {(1-test_size)*100:.0f}% | 测试集 {test_size*100:.0f}%")
        print(f"🎲 随机种子：{random_state}")
    
    def create_directories(self):
        """创建必要的目录结构"""
        print("\n📁 创建目录结构...")
        
        directories = [
            self.yolo_dataset_dir,
            self.train_images_dir,
            self.val_images_dir,
            self.results_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ 创建目录: {directory}")
    
    def load_and_validate_data(self):
        """加载和验证数据"""
        print("\n📂 加载和验证数据...")
        
        # 检查标签文件
        if not os.path.exists(self.labels_file):
            raise FileNotFoundError(f"标签文件不存在: {self.labels_file}")
        
        # 加载标签
        df = pd.read_csv(self.labels_file)
        print(f"📊 标签文件包含 {len(df)} 条记录")
        
        # 检查图像文件
        available_images = []
        missing_images = []
        
        for filename in df['filename'].unique():
            image_path = os.path.join(self.dataset_dir, filename)
            if os.path.exists(image_path):
                available_images.append(filename)
            else:
                missing_images.append(filename)
        
        print(f"✅ 可用图像: {len(available_images)} 张")
        if missing_images:
            print(f"⚠️  缺失图像: {len(missing_images)} 张")
            # 过滤掉缺失的图像
            df = df[df['filename'].isin(available_images)]
        
        print(f"📊 最终有效数据: {len(df)} 条记录")
        return df
    
    def split_dataset(self, df):
        """划分数据集"""
        print(f"\n✂️ 划分数据集...")
        
        # 按文件名分组，确保同一图像的所有标注都在同一个集合中
        unique_files = df['filename'].unique()
        
        # 划分文件列表
        train_files, val_files = train_test_split(
            unique_files, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # 根据文件列表创建训练集和验证集数据
        train_df = df[df['filename'].isin(train_files)].copy()
        val_df = df[df['filename'].isin(val_files)].copy()
        
        print(f"📊 训练集: {len(train_files)} 张图像, {len(train_df)} 条标注")
        print(f"📊 验证集: {len(val_files)} 张图像, {len(val_df)} 条标注")
        
        return train_df, val_df, train_files, val_files
    
    def copy_images(self, train_files, val_files):
        """复制图像到相应目录"""
        print("\n📋 复制图像文件...")
        
        # 复制训练集图像
        print("复制训练集图像...")
        for filename in train_files:
            src = os.path.join(self.dataset_dir, filename)
            dst = os.path.join(self.train_images_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        # 复制验证集图像
        print("复制验证集图像...")
        for filename in val_files:
            src = os.path.join(self.dataset_dir, filename)
            dst = os.path.join(self.val_images_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        print(f"✅ 图像复制完成")
        print(f"   训练集图像: {len(os.listdir(self.train_images_dir))} 张")
        print(f"   验证集图像: {len(os.listdir(self.val_images_dir))} 张")
    
    def create_yolo_labels(self, train_df, val_df):
        """创建YOLO格式的标签文件"""
        print("\n🏷️ 创建YOLO标签文件...")
        
        # 创建标签目录
        train_labels_dir = os.path.join(self.yolo_dataset_dir, 'labels', 'train')
        val_labels_dir = os.path.join(self.yolo_dataset_dir, 'labels', 'val')
        
        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(val_labels_dir, exist_ok=True)
        
        def convert_to_yolo_format(df, labels_dir, image_dir):
            """将边界框转换为YOLO格式"""
            created_files = 0
            
            for filename in df['filename'].unique():
                # 获取该图像的所有标注
                image_annotations = df[df['filename'] == filename]
                
                # 读取图像尺寸
                image_path = os.path.join(image_dir, filename)
                if not os.path.exists(image_path):
                    continue
                
                try:
                    import cv2
                    img = cv2.imread(image_path)
                    if img is None:
                        continue
                    img_height, img_width = img.shape[:2]
                except:
                    # 如果cv2读取失败，使用默认尺寸或跳过
                    continue
                
                # 创建YOLO标签文件
                label_filename = filename.replace('.jpg', '.txt').replace('.png', '.txt')
                label_path = os.path.join(labels_dir, label_filename)
                
                with open(label_path, 'w') as f:
                    for _, row in image_annotations.iterrows():
                        # 获取边界框坐标
                        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                        
                        # 转换为YOLO格式 (center_x, center_y, width, height)，归一化
                        center_x = ((xmin + xmax) / 2) / img_width
                        center_y = ((ymin + ymax) / 2) / img_height
                        width = (xmax - xmin) / img_width
                        height = (ymax - ymin) / img_height
                        
                        # 类别ID (0 = meter)
                        class_id = 0
                        
                        # 写入YOLO格式: class_id center_x center_y width height
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                
                created_files += 1
            
            return created_files
        
        # 为训练集创建标签
        train_labels = convert_to_yolo_format(train_df, train_labels_dir, self.train_images_dir)
        print(f"✅ 训练集标签: {train_labels} 个文件")
        
        # 为验证集创建标签
        val_labels = convert_to_yolo_format(val_df, val_labels_dir, self.val_images_dir)
        print(f"✅ 验证集标签: {val_labels} 个文件")
        
        print("✅ YOLO标签文件创建完成")
    
    def create_data_yaml(self):
        """创建YOLO数据配置文件"""
        print("\n📝 创建数据配置文件...")
        
        # 获取绝对路径避免路径问题
        import os
        abs_dataset_path = os.path.abspath(self.yolo_dataset_dir)
        
        data_yaml_content = f"""path: {abs_dataset_path}
train: images/train
val: images/val
names:
  0: meter
"""
        
        with open('data.yaml', 'w', encoding='utf-8') as f:
            f.write(data_yaml_content)
        
        print(f"✅ data.yaml 文件创建完成")
        print(f"📍 数据集路径: {abs_dataset_path}")
    
    def run_yolo_training(self):
        """运行YOLO模型训练"""
        print("\n🎯 开始YOLO模型训练...")
        
        try:
            # 检查是否安装了ultralytics
            import ultralytics
            print("✅ ultralytics 包已安装")
        except ImportError:
            print("❌ 缺少 ultralytics 包，正在尝试安装...")
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics'], 
                             check=True, capture_output=True)
                print("✅ ultralytics 安装成功")
            except subprocess.CalledProcessError as e:
                print(f"❌ ultralytics 安装失败: {e}")
                return False, "ultralytics 安装失败"
        
        print("执行YOLO训练命令...")
        
        # 方法1：尝试直接使用yolo命令
        yolo_cmd_direct = [
            'yolo', 'train',
            'data=data.yaml',
            'model=yolov8n.pt',
            'epochs=50',
            'imgsz=640', 
            'batch=16',
            'name=meter_detection'
        ]
        
        # 方法2：使用Python API
        yolo_python_script = '''
import os
from ultralytics import YOLO

# 创建模型
model = YOLO('yolov8n.pt')

# 训练模型
results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='meter_detection'
)
print("YOLO训练完成!")
'''
        
        try:
            # 先尝试直接yolo命令
            print("尝试使用 yolo 命令...")
            result = subprocess.run(yolo_cmd_direct, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=1800)
            
            if result.returncode == 0:
                print("✅ YOLO训练完成成功!")
                print("YOLO训练输出:")
                print(result.stdout[-500:])  # 只显示最后500字符
                
                # 检查模型文件是否生成
                yolo_model_path = "runs/detect/meter_detection/weights/best.pt"
                if os.path.exists(yolo_model_path):
                    print(f"✅ YOLO模型已保存: {yolo_model_path}")
                else:
                    print("⚠️  YOLO模型文件未找到，但训练似乎成功")
                
                return True, result.stdout
            else:
                print("❌ yolo 命令失败，尝试使用Python API...")
                print("错误信息:", result.stderr)
                
                # 方法2：使用Python API
                try:
                    # 创建临时脚本文件
                    temp_script = "temp_yolo_train.py"
                    with open(temp_script, 'w') as f:
                        f.write(yolo_python_script)
                    
                    print("使用Python API进行YOLO训练...")
                    result_api = subprocess.run([sys.executable, temp_script], 
                                              capture_output=True, 
                                              text=True, 
                                              timeout=1800)
                    
                    # 删除临时文件
                    if os.path.exists(temp_script):
                        os.remove(temp_script)
                    
                    if result_api.returncode == 0:
                        print("✅ YOLO训练完成成功! (Python API)")
                        print("YOLO训练输出:")
                        print(result_api.stdout[-500:])
                        
                        # 检查模型文件
                        yolo_model_path = "runs/detect/meter_detection/weights/best.pt"
                        if os.path.exists(yolo_model_path):
                            print(f"✅ YOLO模型已保存: {yolo_model_path}")
                        else:
                            print("⚠️  YOLO模型文件未找到，但训练似乎成功")
                        
                        return True, result_api.stdout
                    else:
                        print("❌ Python API 也失败了!")
                        print("错误信息:")
                        print(result_api.stderr)
                        return False, result_api.stderr
                        
                except Exception as api_error:
                    print(f"❌ Python API 出错: {str(api_error)}")
                    return False, str(api_error)
                
        except subprocess.TimeoutExpired:
            print("❌ YOLO训练超时!")
            return False, "YOLO训练超时"
        except Exception as e:
            print(f"❌ YOLO训练出错: {str(e)}")
            return False, str(e)
    
    def run_training(self):
        """运行训练脚本"""
        print("\n🏋️ 开始训练...")
        
        if not os.path.exists(self.train_script):
            raise FileNotFoundError(f"训练脚本不存在: {self.train_script}")
        
        print(f"执行训练脚本: {self.train_script}")
        
        try:
            # 执行训练脚本
            result = subprocess.run([sys.executable, self.train_script], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=3600)  # 1小时超时
            
            if result.returncode == 0:
                print("✅ 训练完成成功!")
                print("训练输出:")
                print(result.stdout)
                return True, result.stdout
            else:
                print("❌ 训练失败!")
                print("错误信息:")
                print(result.stderr)
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            print("❌ 训练超时!")
            return False, "训练超时"
        except Exception as e:
            print(f"❌ 训练出错: {str(e)}")
            return False, str(e)
    
    def run_testing(self):
        """运行测试脚本"""
        print("\n🧪 开始测试...")
        
        if not os.path.exists(self.test_script):
            raise FileNotFoundError(f"测试脚本不存在: {self.test_script}")
        
        print(f"执行测试脚本: {self.test_script}")
        
        try:
            # 执行测试脚本
            result = subprocess.run([sys.executable, self.test_script], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=1800)  # 30分钟超时
            
            if result.returncode == 0:
                print("✅ 测试完成成功!")
                print("测试输出:")
                print(result.stdout)
                return True, result.stdout
            else:
                print("❌ 测试失败!")
                print("错误信息:")
                print(result.stderr)
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            print("❌ 测试超时!")
            return False, "测试超时"
        except Exception as e:
            print(f"❌ 测试出错: {str(e)}")
            return False, str(e)
    
    def extract_metrics_from_output(self, output):
        """从输出中提取性能指标"""
        metrics = {}
        
        # 尝试提取常见的性能指标
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if '准确率' in line or 'accuracy' in line.lower():
                # 尝试提取数字
                import re
                numbers = re.findall(r'\d+\.?\d*%?', line)
                if numbers:
                    metrics['accuracy'] = numbers[-1]
            
            if '损失' in line or 'loss' in line.lower():
                import re
                numbers = re.findall(r'\d+\.?\d+', line)
                if numbers:
                    metrics['loss'] = float(numbers[-1])
            
            if 'F1' in line or 'f1' in line.lower():
                import re
                numbers = re.findall(r'\d+\.?\d+', line)
                if numbers:
                    metrics['f1_score'] = float(numbers[-1])
        
        return metrics
    
    def generate_report(self, train_success, train_output, test_success, test_output, yolo_success=None, yolo_output=None):
        """生成实验报告"""
        print("\n📊 生成实验报告...")
        
        report = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'test_size': self.test_size,
                'random_state': self.random_state,
                'dataset_dir': self.dataset_dir,
                'yolo_dataset_dir': self.yolo_dataset_dir
            },
            'yolo_training': {
                'success': yolo_success if yolo_success is not None else False,
                'output': yolo_output if yolo_success else None,
                'error': yolo_output if yolo_success is False else None,
                'metrics': self.extract_metrics_from_output(yolo_output) if yolo_success and yolo_output else {}
            },
            'training': {
                'success': train_success,
                'output': train_output if train_success else None,
                'error': train_output if not train_success else None,
                'metrics': self.extract_metrics_from_output(train_output) if train_success else {}
            },
            'testing': {
                'success': test_success,
                'output': test_output if test_success else None,
                'error': test_output if not test_success else None,
                'metrics': self.extract_metrics_from_output(test_output) if test_success else {}
            }
        }
        
        # 保存报告
        with open(self.report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 实验报告已保存: {self.report_file}")
        
        # 打印摘要
        print("\n" + "="*60)
        print("📊 实验摘要")
        print("="*60)
        print(f"⏰ 实验时间: {report['experiment_info']['timestamp']}")
        print(f"📊 数据划分: 训练集 {(1-self.test_size)*100:.0f}% | 测试集 {self.test_size*100:.0f}%")
        print(f"🎯 YOLO训练: {'✅ 成功' if yolo_success else '❌ 失败' if yolo_success is False else '⏭️ 跳过'}")
        print(f"🏋️ CRNN训练: {'✅ 成功' if train_success else '❌ 失败'}")
        print(f"🧪 测试状态: {'✅ 成功' if test_success else '❌ 失败'}")
        
        if yolo_success and report['yolo_training']['metrics']:
            print("📈 YOLO指标:")
            for key, value in report['yolo_training']['metrics'].items():
                print(f"   {key}: {value}")
        
        if train_success and report['training']['metrics']:
            print("📈 CRNN指标:")
            for key, value in report['training']['metrics'].items():
                print(f"   {key}: {value}")
        
        if test_success and report['testing']['metrics']:
            print("📈 测试指标:")
            for key, value in report['testing']['metrics'].items():
                print(f"   {key}: {value}")
        
        print("="*60)
        
        return report
    
    def run_full_pipeline(self, skip_yolo=False):
        """运行完整的训练测试流程"""
        print("🎯 开始完整的自动化训练测试流程")
        print("="*60)
        
        try:
            # 步骤1: 创建目录
            self.create_directories()
            
            # 步骤2: 加载和验证数据
            df = self.load_and_validate_data()
            
            # 步骤3: 划分数据集
            train_df, val_df, train_files, val_files = self.split_dataset(df)
            
            # 步骤4: 复制图像
            self.copy_images(train_files, val_files)
            
            # 步骤5: 创建YOLO标签文件
            self.create_yolo_labels(train_df, val_df)
            
            # 步骤6: 创建配置文件
            self.create_data_yaml()
            
            # 步骤7: 运行YOLO训练
            yolo_success, yolo_output = None, None
            if not skip_yolo:
                yolo_success, yolo_output = self.run_yolo_training()
            else:
                print("⏭️ 跳过YOLO训练步骤")
            
            # 步骤8: 运行CRNN训练
            train_success, train_output = self.run_training()
            
            # 步骤9: 运行测试
            test_success, test_output = self.run_testing()
            
            # 步骤10: 生成报告
            report = self.generate_report(train_success, train_output, test_success, test_output, yolo_success, yolo_output)
            
            print("\n🎉 自动化流程完成!")
            print(f"📊 详细报告请查看: {self.report_file}")
            
            return report
            
        except Exception as e:
            print(f"\n❌ 流程执行失败: {str(e)}")
            raise

def main():
    """主函数"""
    print("🚀 CRNN模型自动化训练测试系统")
    print("=" * 60)
    
    # 创建自动训练测试实例
    auto_trainer = AutoTrainTest(
        test_size=0.2,      # 20%作为测试集
        random_state=42     # 随机种子确保可重复性
    )
    
    # 运行完整流程
    try:
        report = auto_trainer.run_full_pipeline()
        
        print("\n✅ 自动化流程执行完成!")
        
        # 检查是否都成功
        yolo_ok = report['yolo_training']['success'] if report['yolo_training']['success'] is not None else True
        if report['training']['success'] and report['testing']['success'] and yolo_ok:
            print("🎉 YOLO训练、CRNN训练和测试都成功完成!")
            return 0
        else:
            print("⚠️  部分流程未成功完成，请查看详细报告")
            return 1
            
    except Exception as e:
        print(f"\n❌ 自动化流程失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main()) 
#!/usr/bin/env python3
"""
对比预测结果与真实标签，计算准确率
"""

import pandas as pd
import os

# 配置参数
PREDICTIONS_FILE = './results.csv'
LABELS_FILE = '/Volumes/YMM/Dataset/labels.csv'

def evaluate_predictions():
    """对比预测结果与真实标签"""
    print("🔍 开始评估预测结果...")
    print("=" * 60)
    
    # 检查文件是否存在
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"❌ 预测结果文件不存在: {PREDICTIONS_FILE}")
        return
    
    if not os.path.exists(LABELS_FILE):
        print(f"❌ 标签文件不存在: {LABELS_FILE}")
        return
    
    try:
        # 读取预测结果 (没有表头)
        predictions_df = pd.read_csv(PREDICTIONS_FILE, header=None, names=['id', 'predicted_reading'])
        print(f"✅ 预测结果加载成功，共 {len(predictions_df)} 条记录")
        
        # 读取真实标签
        labels_df = pd.read_csv(LABELS_FILE)
        print(f"✅ 标签文件加载成功，共 {len(labels_df)} 条记录")
        
    except Exception as e:
        print(f"❌ 文件读取失败: {e}")
        return
    
    # 检查标签文件是否包含number列
    if 'number' not in labels_df.columns:
        print(f"❌ 标签文件中没有找到 'number' 列")
        print(f"   可用列: {list(labels_df.columns)}")
        return
    
    # 确保两个数据集的长度一致
    min_length = min(len(predictions_df), len(labels_df))
    if len(predictions_df) != len(labels_df):
        print(f"⚠️  数据集长度不一致: 预测结果 {len(predictions_df)} vs 标签 {len(labels_df)}")
        print(f"   将使用前 {min_length} 条记录进行比较")
    
    # 截取相同长度的数据
    predictions_df = predictions_df.head(min_length)
    labels_df = labels_df.head(min_length)
    
    # 提取真实读数
    true_readings = labels_df['number'].astype(str)
    predicted_readings = predictions_df['predicted_reading'].astype(str)
    
    # 计算准确率
    correct_predictions = 0
    total_predictions = len(predictions_df)
    
    # 详细比较结果
    comparison_results = []
    
    for i in range(total_predictions):
        true_val = true_readings.iloc[i]
        pred_val = predicted_readings.iloc[i]
        
        # 精确匹配
        is_correct = (true_val == pred_val)
        
        if is_correct:
            correct_predictions += 1
        
        comparison_results.append({
            'id': i + 1,
            'true_reading': true_val,
            'predicted_reading': pred_val,
            'correct': is_correct
        })
    
    # 计算准确率
    accuracy = correct_predictions / total_predictions * 100
    
    # 显示结果
    print(f"\n📊 评估结果:")
    print(f"   总样本数: {total_predictions}")
    print(f"   预测正确: {correct_predictions}")
    print(f"   预测错误: {total_predictions - correct_predictions}")
    print(f"   准确率: {accuracy:.2f}%")
    
    # 显示一些错误案例
    incorrect_cases = [case for case in comparison_results if not case['correct']]
    
    if incorrect_cases:
        print(f"\n❌ 错误案例 (前10个):")
        for i, case in enumerate(incorrect_cases[:10]):
            print(f"   ID {case['id']}: 真实={case['true_reading']}, 预测={case['predicted_reading']}")
    
    # 显示一些正确案例
    correct_cases = [case for case in comparison_results if case['correct']]
    
    if correct_cases:
        print(f"\n✅ 正确案例 (前5个):")
        for i, case in enumerate(correct_cases[:5]):
            print(f"   ID {case['id']}: 真实={case['true_reading']}, 预测={case['predicted_reading']}")
    
    # 保存详细比较结果
    comparison_df = pd.DataFrame(comparison_results)
    comparison_output = './comparison_results.csv'
    comparison_df.to_csv(comparison_output, index=False)
    print(f"\n💾 详细比较结果已保存到: {comparison_output}")
    
    # 数值分析 (尝试将读数转换为数值进行更详细的分析)
    print(f"\n🔢 数值分析:")
    try:
        true_numeric = pd.to_numeric(true_readings, errors='coerce')
        pred_numeric = pd.to_numeric(predicted_readings, errors='coerce')
        
        # 过滤掉无法转换为数值的数据
        valid_mask = ~(true_numeric.isna() | pred_numeric.isna())
        true_numeric_valid = true_numeric[valid_mask]
        pred_numeric_valid = pred_numeric[valid_mask]
        
        if len(true_numeric_valid) > 0:
            # 计算平均绝对误差
            mae = abs(true_numeric_valid - pred_numeric_valid).mean()
            print(f"   有效数值样本: {len(true_numeric_valid)}")
            print(f"   平均绝对误差 (MAE): {mae:.4f}")
            
            # 计算在一定误差范围内的准确率
            tolerance_levels = [0.1, 0.5, 1.0]
            for tol in tolerance_levels:
                within_tolerance = abs(true_numeric_valid - pred_numeric_valid) <= tol
                tolerance_accuracy = within_tolerance.sum() / len(true_numeric_valid) * 100
                print(f"   误差 ≤ {tol} 的准确率: {tolerance_accuracy:.2f}%")
        else:
            print("   无有效的数值数据进行分析")
            
    except Exception as e:
        print(f"   数值分析失败: {e}")

if __name__ == '__main__':
    evaluate_predictions() 
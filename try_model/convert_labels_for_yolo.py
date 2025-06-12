import os
import csv
from PIL import Image

# 输入输出目录
INPUT_CSV = './Dataset/labels.csv'
IMAGES_DIR = './Dataset'
OUTPUT_DIR = './yolov8_dataset/labels'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_image_size(image_path):
    """获取图片实际尺寸"""
    with Image.open(image_path) as img:
        return img.size

def normalize_bbox(xmin, ymin, xmax, ymax, img_width, img_height):
    """标准化边界框坐标到[0,1]范围，并修正超出边界的坐标"""
    # 修正超出边界的坐标
    xmin = max(0, min(xmin, img_width))
    ymin = max(0, min(ymin, img_height))
    xmax = max(xmin, min(xmax, img_width))
    ymax = max(ymin, min(ymax, img_height))
    
    # 计算YOLO格式坐标
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    w = (xmax - xmin) / img_width
    h = (ymax - ymin) / img_height
    
    return (x_center, y_center, w, h)

error_count = 0
total_count = 0

with open(INPUT_CSV, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过标题行
    for fn, num, xmin, ymin, xmax, ymax in reader:
        total_count += 1
        
        # 获取图片实际尺寸
        img_path = os.path.join(IMAGES_DIR, fn)
        if not os.path.exists(img_path):
            print(f"⚠️  图片不存在: {fn}")
            continue
            
        img_width, img_height = get_image_size(img_path)
        
        # 转换坐标
        xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
        
        # 检查是否需要修正
        if xmin < 0 or ymin < 0 or xmax > img_width or ymax > img_height:
            error_count += 1
            print(f"⚠️  修正边界框: {fn} - 原始: ({xmin},{ymin},{xmax},{ymax}) -> 图片尺寸: {img_width}x{img_height}")
        
        yolo_label = normalize_bbox(xmin, ymin, xmax, ymax, img_width, img_height)
        
        # 验证结果
        if any(coord < 0 or coord > 1 for coord in yolo_label):
            print(f"❌ 标准化后仍有问题: {fn} -> {yolo_label}")
        
        txt_path = os.path.join(OUTPUT_DIR, fn.replace('.jpg','.txt'))
        with open(txt_path, 'w') as out_f:
            out_f.write(f"0 {' '.join(map(str, yolo_label))}\n")

print(f"✅ 标签转换完成！总计: {total_count}, 修正: {error_count}")

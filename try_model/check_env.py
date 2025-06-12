import sys
import torch
import torchvision
import PIL
import numpy as np

# 检查 OpenCV (可选)
try:
    import cv2
    opencv_version = cv2.__version__
except ImportError:
    opencv_version = None
except Exception as e:
    opencv_version = f"错误: {e}"

# 检查 ultralytics (YOLOv8)
try:
    import ultralytics
    yolo_version = ultralytics.__version__
except ImportError:
    yolo_version = None

print("="*50)
print("🚀 电表读数识别项目 环境检测工具")
print("="*50)

# Python版本
print(f"Python 版本: {sys.version.split()[0]}")

# Torch版本及GPU支持
print(f"PyTorch 版本: {torch.__version__}")
print(f"GPU 是否可用: {'✅ 可用' if torch.cuda.is_available() else '❌ 不可用'}")
if torch.cuda.is_available():
    print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")

# torchvision
print(f"torchvision 版本: {torchvision.__version__}")

# PIL
print(f"Pillow 版本: {PIL.__version__}")

# numpy
print(f"numpy 版本: {np.__version__}")

# opencv (可选)
if opencv_version:
    if opencv_version.startswith("错误"):
        print(f"❌ OpenCV: {opencv_version}")
        print("   注意: OpenCV不是必需的，可以忽略此错误")
    else:
        print(f"opencv 版本: {opencv_version}")
else:
    print("⚠️  OpenCV 未安装（项目不需要，可忽略）")

# ultralytics
if yolo_version:
    print(f"ultralytics (YOLOv8) 版本: {yolo_version}")
else:
    print("❌ 未安装 ultralytics，请运行： pip install ultralytics")

print("="*50)
print("✅ 检测完成，请确认全部正常")

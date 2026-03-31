from ultralytics import YOLO

# 1. 加载你的优等生模型
model = YOLO('best.pt')

# 2. 导出为 ONNX 格式
# format='onnx' 指定格式
# opset=12 保证兼容性
# simplify=True 会自动帮你精简模型结构，让它更小、更快
path = model.export(format='onnx', opset=12, simplify=True)

print(f"转换成功！模型文件保存在: {path}")
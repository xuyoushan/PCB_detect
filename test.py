from ultralytics import YOLO

# 加载模型
model = YOLO('best.pt')

# 打印模型能识别的所有标签
print("这个模型可以识别以下物体：")
print(model.names)
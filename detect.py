# from ultralytics import YOLO
# import cv2
#
# # 加载模型
# model = YOLO('best.pt')
#
# # 读取图片
# img = cv2.imread('test.jpg')
# h, w, _ = img.shape
#
# # 这种图左边是模板，右边是缺陷。我们只给模型看右半部分
# # 如果你不想裁剪，也可以直接用全图
# test_img = img[:, w//2:]
#
# # 运行预测
# # conf=0.15: 降低门槛，让模型更“敏锐”
# # iou=0.5: 防止同一个瑕疵画多个框
# results = model.predict(source=test_img, save=True, conf=0.45, iou=0.5)
#
# for result in results:
#     if len(result.boxes) == 0:
#         print("依然没有检测到缺陷。请尝试进一步降低 conf 阈值。")
#     else:
#         print(f"成功！检测到 {len(result.boxes)} 个缺陷。")
#         print(result.boxes)
#
# print("结果已保存至 runs/detect/predict")

from ultralytics import YOLO
import cv2

# 1. 加载你刚刚转换成功的 ONNX 模型
# task='detect' 明确告诉它这是目标检测任务
model = YOLO('best.onnx', task='detect')

# 2. 读取图片并只截取右半部分（针对你那张 Kaggle 双拼图）
img = cv2.imread('test.jpg')
if img is not None:
    h, w, _ = img.shape
    # 裁剪右半边：这是有缺陷的部分
    test_img = img[:, w // 2:]

    # 3. 使用 ONNX 模型进行推理
    # save=True 会自动帮你把画好框的图存在 runs/detect/predict 文件夹下
    results = model.predict(source=test_img, save=True, conf=0.25)

    print("★ ONNX 推理成功！")
    print(f"结果保存在最新的 runs/detect 文件夹中。")
else:
    print("错误：找不到 test.jpg，请检查文件名和路径。")
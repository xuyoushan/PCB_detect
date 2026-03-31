import time
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import cv2
import numpy as np
import matplotlib

# === 关键修正 1：强行切换绘图后端，避开 PyCharm 的 Bug ===
try:
    matplotlib.use('TkAgg')
except:
    pass

# 设置中文显示（解决 PPT 截图乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def get_performance(display_name, model_path, img_path="test_pcb_sample.jpg"):
    print(f"正在准备测试: {display_name}...")
    try:
        model = YOLO(model_path)
        # 预热
        for _ in range(3):
            model.predict(img_path, device='cpu', verbose=False)
        # 测 20 次
        start_time = time.time()
        for _ in range(20):
            model.predict(img_path, device='cpu', verbose=False)
        avg_time = ((time.time() - start_time) / 20) * 1000
        return avg_time
    except:
        print(f"❌ {display_name} 下载/加载失败，已跳过。")
        return None


if __name__ == "__main__":
    test_img = "test_pcb_sample.jpg"
    if not os.path.exists(test_img):
        cv2.imwrite(test_img, np.zeros((640, 640, 3), dtype=np.uint8))

    # 丰富了对比模型阵列
    models_to_compare = {
        "Baseline\n(YOLOv5n)": "yolov5nu.pt",
        "Standard-Lite\n(YOLOv8n)": "yolov8n.pt",
        "Industrial-Base\n(YOLOv8s)": "yolov8s.pt",
        "Real-Time SOTA\n(YOLOv10n)": "yolov10n.pt",
        "Latest-Gen\n(YOLOv11n)": "yolov11n.pt",
        "智绘精芯\n(Ours/Modified)": "best.pt"
    }

    results_data = {}
    for name, path in models_to_compare.items():
        # 如果是本地权重 best.pt，检查文件
        if path == "best.pt" and not os.path.exists(path):
            continue
        latency = get_performance(name, path, test_img)
        if latency:
            results_data[name] = latency

    if results_data:
        # 1. 数据准备与名称映射 (为了让你看懂，这里做了双行显示处理)
        # \n 代表换行，这样版本号会出现在功能定义下方
        names = list(results_data.keys())
        times = list(results_data.values())

        # 2. 设置画布：调高底边距 (bottom=0.2) 解决文字重合问题
        fig, ax = plt.subplots(figsize=(12, 7.5), dpi=120)
        plt.subplots_adjust(bottom=0.22)

        # 3. 配色方案：多彩高饱和
        vibrant_colors = ['#FF6B6B', '#FF9F43', '#A55EEA', '#26DE81', '#45AAF2', '#2D3436']
        current_colors = vibrant_colors[:len(names) - 1] + [vibrant_colors[-1]]

        # 4. 绘制柱状图
        bars = ax.bar(names, times, color=current_colors, width=0.42, edgecolor='white', linewidth=1.2, zorder=3)

        # 5. 工业实时性阈值线 (200ms)
        threshold = 200
        plt.axhline(y=threshold, color='#E63946', linestyle='--', linewidth=2, alpha=0.6, zorder=2)
        plt.text(len(names) - 0.5, threshold + 6, f"工业级实时阈值 ({threshold}ms)",
                 color='#E63946', ha='right', va='bottom', fontsize=10, fontweight='bold')

        # 6. 数值标注
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 3,
                    f'{height:.1f} ms', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color=bar.get_facecolor())

        # 7. 标题与坐标轴优化：labelpad 解决重叠关键
        plt.title("“智绘精芯”系统推理性能与工业标准对标实测", fontsize=18, fontweight='bold', pad=35)
        plt.ylabel("平均推理响应时间 (ms)", fontsize=12, fontweight='bold')

        # 增加 labelpad，让“模型架构”往下挪，不跟下方的注释打架
        plt.xlabel("模型架构与版本对标", fontsize=12, fontweight='bold')

        # 8. 细节打磨
        ax.grid(axis='y', ls=':', alpha=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 9. 底部专业注释
        # y 坐标调低到 0.02，确保在最底部
        plt.figtext(0.5, 0.01,
                    f"* 实验环境: CPU (Intel/AMD) | 样本尺寸: 640x640 | 调度策略: 20-Round Iterative Average",
                    fontsize=9, color='#7f8c8d', style='italic', ha='center')

        plt.tight_layout()
        print("\n[可视化优化完成] 标签已更新，重叠问题已解决。")
        plt.show()
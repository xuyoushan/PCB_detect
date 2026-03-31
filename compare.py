import time
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import matplotlib
matplotlib.use('TkAgg') # 强制使用独立的窗口模式，避开 PyCharm 的 bug

# 解决画图中文显示问题（针对 Windows 系统）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def run_benchmark():
    # 1. 加载模型
    print("正在加载模型...")
    model_official = YOLO('yolov8s.pt')
    model_mine = YOLO('best.pt')

    img_path = 'test.jpg'
    iterations = 10

    def get_time(model, name):
        print(f"开始测试 {name}...")
        # 预热处理
        model.predict(source=img_path, conf=0.25, save=False, verbose=False)

        start = time.time()
        for _ in range(iterations):
            model.predict(source=img_path, conf=0.25, save=False, verbose=False)
        end = time.time()

        avg_ms = ((end - start) / iterations) * 1000
        return avg_ms

    # 2. 执行测试获取数据
    t_official = get_time(model_official, "官方原版 YOLOv8s")
    t_mine = get_time(model_mine, "改进轻量化模型")

    # 3. 开始绘图
    labels = ['官方原版 (YOLOv8s)', '改进轻量化 (YOLO-RLGI)']
    times = [t_official, t_mine]
    colors = ['#d62728', '#2ca02c']  # 红色和绿色

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, times, color=colors, width=0.6)

    # 在柱状图上方标注具体数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 2, f'{yval:.2f} ms',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.ylabel('平均推理耗时 (ms)', fontsize=12)
    plt.title('模型推理速度对比测试 (CPU环境)', fontsize=15)
    plt.ylim(0, max(times) * 1.2)  # 留出顶部空间显示文字

    # 添加对比结论文字
    improvement = ((t_official - t_mine) / t_official) * 100
    plt.figtext(0.5, 0.01, f"实验结论：改进版模型推理速度提升了约 {improvement:.1f}%",
                ha="center", fontsize=12, bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})

    # 4. 保存并显示
    plt.tight_layout()
    plt.savefig('comparison_chart.png', dpi=300)
    print(f"\n★ 统计图已生成并保存为: comparison_chart.png")
    print(f"★ 官方耗时: {t_official:.2f}ms, 改进版耗时: {t_mine:.2f}ms")
    print("正在生成对比结果图...")
    # 官方模型跑一张，保存结果
    model_official.predict(source=img_path, save=True, line_width=2, project="runs/compare", name="official")
    # 你的模型跑一张，保存结果
    model_mine.predict(source=img_path, save=True, line_width=2, project="runs/compare", name="mine")
    plt.show()


if __name__ == '__main__':
    run_benchmark()
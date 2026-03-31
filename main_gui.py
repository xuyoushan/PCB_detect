import sys
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ultralytics import YOLO


class ScienceFictionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智绘精芯 - PCB工业缺陷智能检测系统 v2.0")
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("icon.png"))  # 建议找个芯片图标

        # 加载你的模型
        self.model = YOLO('best.pt')

        self.initUI()
        self.setStyleSheet(self.get_qss())

    def initUI(self):
        # 主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # --- 左侧控制面板 ---
        left_panel = QFrame()
        left_panel.setFixedWidth(280)
        left_panel.setObjectName("ControlPanel")
        left_layout = QVBoxLayout(left_panel)

        title = QLabel("SYSTEM CONTROL")
        title.setObjectName("PanelTitle")
        left_layout.addWidget(title)

        self.btn_load = QPushButton(" 📂 载入待检图像")
        self.btn_load.clicked.connect(self.load_image)
        left_layout.addWidget(self.btn_load)

        self.btn_detect = QPushButton(" ⚡ 开启核心识别")
        self.btn_detect.clicked.connect(self.start_detection)
        self.btn_detect.setEnabled(False)
        left_layout.addWidget(self.btn_detect)

        left_layout.addStretch()

        # 数据显示区
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setObjectName("LogBox")
        self.info_box.setPlaceholderText("系统就绪，等待指令...")
        left_layout.addWidget(QLabel("实时分析数据流:"))
        left_layout.addWidget(self.info_box)

        # --- 右侧展示区 ---
        right_panel = QFrame()
        right_panel.setObjectName("DisplayPanel")
        right_layout = QVBoxLayout(right_panel)

        self.img_label = QLabel("STANDBY - WAITING FOR INPUT")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setObjectName("ImageDisplay")
        right_layout.addWidget(self.img_label)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        right_layout.addWidget(self.progress)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

    def get_qss(self):
        return """
        QMainWindow { background-color: #050A10; }
        #ControlPanel { 
            background-color: rgba(15, 25, 45, 200); 
            border: 2px solid #00F2FF;
            border-radius: 10px;
        }
        #DisplayPanel { 
            background-color: #000000; 
            border: 1px solid #1A3050;
            background-image: url('grid.png'); /* 建议弄个透明网格背景 */
        }
        QLabel { color: #00F2FF; font-family: "Segoe UI Semibold"; }
        #PanelTitle { font-size: 20px; font-weight: bold; margin-bottom: 10px; border-bottom: 1px solid #00F2FF; }
        QPushButton {
            background-color: #1A3050;
            color: #00F2FF;
            border: 1px solid #00F2FF;
            padding: 12px;
            font-size: 14px;
            border-radius: 5px;
        }
        QPushButton:hover { background-color: #00F2FF; color: #000000; }
        QPushButton:disabled { border-color: #333; color: #555; }
        #LogBox { 
            background-color: #050A10; 
            color: #00FF41; 
            border: 1px solid #1A3050; 
            font-family: 'Consolas';
        }
        QProgressBar {
            border: 1px solid #00F2FF;
            background: #050A10;
            text-align: center;
            color: white;
        }
        QProgressBar::chunk { background-color: #00F2FF; }
        """

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择PCB图像', '', 'Images (*.jpg *.png *.jpeg)')
        if fname:
            self.current_img_path = fname
            pixmap = QPixmap(fname)
            self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.btn_detect.setEnabled(True)
            self.log(f">> 载入图像: {fname.split('/')[-1]}")

    def log(self, text):
        self.info_box.append(text)

    def start_detection(self):
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.log(">> 正在初始化 LSKA 注意力引擎...")

        # 模拟扫描动画
        for i in range(1, 101):
            QThread.msleep(10)  # 模拟计算耗时
            self.progress.setValue(i)
            QApplication.processEvents()

        # 执行推理
        results = self.model.predict(source=self.current_img_path, conf=0.25)
        res = results[0]

        # 结果处理
        annotated_frame = res.plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = annotated_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(annotated_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.img_label.setPixmap(
            QPixmap.fromImage(q_img).scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        count = len(res.boxes)
        self.log(f">> 检测完成！发现 {count} 处异常风险点。")
        self.log("-" * 30)
        self.progress.setVisible(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ScienceFictionGUI()
    gui.show()
    sys.exit(app.exec_())
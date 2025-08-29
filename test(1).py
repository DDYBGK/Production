import sys
import numpy as np
import torch
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem
import os
import traceback
import open3d as o3d
import matplotlib as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
    QMessageBox
)
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
from data_utils.indoor3d_util import g_label2color
from PyQt5.QtCore import QThread, pyqtSignal

# Add the current directory to system path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Import test_inference functions
from test_semseg_simple11111 import *

class DataLoaderThread(QThread):
    data_loaded = pyqtSignal(np.ndarray)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            if self.file_path.endswith('.npy'):
                points = np.load(self.file_path)
            elif self.file_path.endswith('.txt'):
                # 假设txt为空格/逗号分隔，自动适应
                points = np.loadtxt(self.file_path, delimiter=None)
            else:
                points = None
            self.data_loaded.emit(points)
        except Exception as e:
            self.data_loaded.emit(None)

class PointCloudViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Point Cloud Viewer")
        self.setGeometry(100, 100, 1200, 700)
        
        # 初始化模型
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.load_train_model()
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Model loading failed: {str(e)}")

        # 主窗口布局
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)

        # 上部分：图像显示（左右）
        self.plot_layout = QHBoxLayout()
        self.main_layout.addLayout(self.plot_layout)

        # # 左侧：原始点云
        # self.original_fig = Figure()
        # self.original_canvas = FigureCanvas(self.original_fig)
        # self.original_ax = self.original_fig.add_subplot(111, projection='3d')
        # self.original_ax.set_title("src pointcloud")
        # self.original_ax.set_axis_off()
        # self.plot_layout.addWidget(self.original_canvas)
        #
        # # 右侧：分割后结果
        # self.result_fig = Figure()
        # self.result_canvas = FigureCanvas(self.result_fig)
        # self.result_ax = self.result_fig.add_subplot(111, projection='3d')
        # self.result_ax.set_title("seg pointcloud")
        # self.result_ax.set_axis_off()
        # self.plot_layout.addWidget(self.result_canvas)
        # 左侧：原始点云视图
        self.original_view = GLViewWidget()
        self.original_view.setBackgroundColor('w')
        self.original_view.setCameraPosition(distance=10)
        self.plot_layout.addWidget(self.original_view)

        # 右侧：分割后结果视图
        self.result_view = GLViewWidget()
        self.result_view.setBackgroundColor('w')
        self.result_view.setCameraPosition(distance=10)
        self.plot_layout.addWidget(self.result_view)

        # 下部分：按钮
        self.button_layout = QHBoxLayout()
        self.main_layout.addLayout(self.button_layout)

        self.load_button = QPushButton("加载文件")
        self.load_button.clicked.connect(self.load_point_cloud)
        self.button_layout.addWidget(self.load_button)

        self.detect_button = QPushButton("点云分割")
        self.detect_button.clicked.connect(self.detect_objects)
        self.button_layout.addWidget(self.detect_button)

        self.clear_button = QPushButton("清除工作区")
        self.clear_button.clicked.connect(self.clear_views)
        self.button_layout.addWidget(self.clear_button)

        # 存储当前加载的点云数据
        self.current_points = None
        self.current_dataset = None
        self.thread = None

    def load_train_model(self):
        """Load the trained model"""
        try:
            self.model = load_model()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def draw_point_cloud(self, view, points, colors=None):
        try:
            # 清除已有图形
            view.items.clear()

            if points is None or len(points) == 0:
                print("No points to display")
                return

            # 设置点云位置
            pos = points[:, :3]

            # 如果没有颜色，则按高度着色
            if colors is None or colors.shape[1] < 3:
                z = pos[:, 2]
                norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
                cmap = plt.get_cmap('viridis')
                colors = cmap(norm)[:, :3]  # 去掉 alpha 通道

            else:
                # 归一化 RGB 到 [0,1]
                colors = np.clip(colors / 255.0, 0, 1)

            # 创建散点图项
            scatter = GLScatterPlotItem(pos=pos, color=colors, size=2, pxMode=True)
            view.addItem(scatter)
            print("Point cloud rendered with OpenGL successfully.")

        except Exception as e:
            print(f"Error drawing point cloud with OpenGL: {e}")
            QMessageBox.warning(self, "Error", f"Failed to render point cloud: {str(e)}")

    def load_point_cloud(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Point Cloud File", "", "Point Cloud Files (*.npy *.txt);;NumPy Files (*.npy);;Text Files (*.txt)"
        )
        if not file_path:
            return
        self.load_button.setEnabled(False)
        self.thread = DataLoaderThread(file_path)
        self.thread.data_loaded.connect(self.on_data_loaded)
        self.thread.start()

    def on_data_loaded(self, points):
        self.load_button.setEnabled(True)
        if points is None or len(points) == 0:
            QMessageBox.warning(self, "Error", "Failed to load or empty point cloud!")
            return
        self.current_points = points

        # 特征补全
        if self.current_points.shape[1] < 9:
            points_with_features = np.zeros((self.current_points.shape[0], 9))
            points_with_features[:, :self.current_points.shape[1]] = self.current_points
            if self.current_points.shape[1] < 6:
                points_with_features[:, 3:6] = 255  # 默认白色
            points_with_features[:, 6:] = 0  # 默认法向量
            self.current_points = points_with_features

        # 用原始文件名保存
        temp_dir = os.path.join(os.path.dirname(__file__), "temp_data")
        os.makedirs(temp_dir, exist_ok=True)
        # 获取原始文件名
        if hasattr(self.thread, 'file_path'):
            orig_name = os.path.basename(self.thread.file_path)
        else:
            orig_name = "temp.npy"
        self.temp_file = os.path.join(temp_dir, orig_name)
        np.save(self.temp_file, self.current_points)

        # 使用 OpenGL 渲染原始点云
        self.draw_point_cloud(self.original_view, self.current_points)

    def detect_objects(self):
        if self.current_points is None:
            QMessageBox.warning(self, "Warning", "Please load point cloud first!")
            return
        
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Model not loaded!")
            return

        try:
            infer_res_save_path = os.path.join(os.path.dirname(__file__), "infer_res")
            os.makedirs(infer_res_save_path, exist_ok=True)
            inference(self.model, self.temp_file,infer_res_save_path)
            
            # 显示结果
            if os.path.exists(infer_res_save_path+"/"+os.path.basename(self.temp_file).replace(".npy", "_pred.txt")):
                colored_points = np.loadtxt(infer_res_save_path+"/"+os.path.basename(self.temp_file).replace(".npy", "_pred.txt"))
                # colored_points 格式是 x y z r g b，取前3个坐标和后3个颜色
                self.draw_point_cloud(self.result_view, colored_points[:, :3], colored_points[:, 3:])
            else:
                colored_points = self.current_points
                self.draw_point_cloud(self.result_view, self.current_points)
            
        except Exception as e:
            print(f"Error during inference: {e}")
            #print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed during inference: {str(e)}")

    def clear_views(self):
        try:
            self.original_ax.clear()
            self.original_ax.set_title("src pointcloud")
            self.original_ax.set_axis_off()
            self.original_canvas.draw()

            self.result_ax.clear()
            self.result_ax.set_title("seg pointcloud")
            self.result_ax.set_axis_off()
            self.result_canvas.draw()
        except Exception as e:
            print(f"Error clearing views: {e}")
            QMessageBox.warning(self, "Error", f"Failed to clear views: {str(e)}")


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        viewer = PointCloudViewer()
        viewer.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application error: {e}")
        print(traceback.format_exc())
        QMessageBox.critical(None, "Fatal Error", f"Application crashed: {str(e)}")

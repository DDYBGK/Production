import sys
import numpy as np
import torch
import os
import traceback
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
    QMessageBox, QProgressBar, QLabel
)
from PyQt5.QtCore import QThread, pyqtSignal
import pyqtgraph.opengl as gl
from PyQt5.QtGui import QPixmap
from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
from data_utils.indoor3d_util import g_label2color

# Add the current directory to system path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Import test_inference functions
from test_semseg_simple import *


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

        # 新增模型类型状态变量
        self.current_model_type = "pointnet2"  # 默认模型A

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
        self.main_layout.setSpacing(5)  # 设置布局间距

        # 新增：图片和标题布局
        image_title_layout = QHBoxLayout()
        self.main_layout.addLayout(image_title_layout)

        # 插入图片
        logo_path = os.path.join(BASE_DIR,  "1.png")
        if os.path.exists(logo_path):
            logo_label = QLabel()
            pixmap = QPixmap(logo_path)
            logo_label.setPixmap(pixmap.scaledToHeight(200))
            image_title_layout.addWidget(logo_label)
        else:
            print(f"Logo image not found at {logo_path}")

        # 添加标题
        title_label = QLabel("面向具身智能机器人的点云语义分割系统")
        title_label.setStyleSheet("font-size: 60px; font-weight: bold;")
        image_title_layout.addWidget(title_label)

        # 上部分：3D显示区域
        self.gl_layout = QHBoxLayout()
        self.main_layout.addLayout(self.gl_layout, 4)  # 设置高度单位为4

        # 左侧：原始点云
        self.original_view = gl.GLViewWidget()
        self.original_view.setCameraPosition(distance=15)
        self.original_g = gl.GLGridItem()
        self.original_view.addItem(self.original_g)
        self.gl_layout.addWidget(self.original_view)

        # 右侧：分割结果
        self.result_view = gl.GLViewWidget()
        self.result_view.setCameraPosition(distance=15)
        self.result_g = gl.GLGridItem()
        self.result_view.addItem(self.result_g)
        self.gl_layout.addWidget(self.result_view)

        # 中间部分：按钮
        self.button_layout = QHBoxLayout()
        self.main_layout.addLayout(self.button_layout, 1)  # 设置高度单位为1

        self.load_button = QPushButton("加载")
        self.load_button.clicked.connect(self.load_point_cloud)
        self.button_layout.addWidget(self.load_button)

        self.detect_button = QPushButton("检测")
        self.detect_button.clicked.connect(self.detect_objects)
        self.button_layout.addWidget(self.detect_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_views)
        self.button_layout.addWidget(self.clear_button)

        # 在按钮布局中添加切换按钮
        self.model_switch_btn = QPushButton("切换模型(pointnet2/LCHNet)")
        self.model_switch_btn.clicked.connect(self.switch_model)
        self.button_layout.addWidget(self.model_switch_btn)

        # 下部分：进度条和准确率显示
        self.progress_layout = QHBoxLayout()
        self.progress_layout.setSpacing(10)
        self.main_layout.addLayout(self.progress_layout, 1)  # 设置高度单位为1

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedHeight(20)  # 固定高度
        self.progress_bar.setValue(0)
        self.progress_layout.addWidget(self.progress_bar)

        self.accuracy_label = QLabel("Accuracy: N/A")
        self.progress_layout.addWidget(self.accuracy_label)

        # 存储当前数据
        self.current_points = None
        self.current_plot = None
        self.result_plot = None
        self.thread = None

    def switch_model(self):
        pointnet2="LCHNet"
        pointnet = "pointnet2"
        """切换模型类型"""
        self.current_model_type = "pointnet" if self.current_model_type == "pointnet2" else "pointnet2"
        try:
            self.load_train_model()
            if self.current_model_type=="pointnet2":
                QMessageBox.information(self, "提示", f"已切换到模型{pointnet2}")
            else:
                QMessageBox.information(self, "提示", f"已切换到模型{pointnet}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型切换失败: {str(e)}")

    def load_train_model(self):
        """根据当前类型加载模型"""
        try:
            self.model = load_model(self.current_model_type)  # 传入模型类型
            print(f"Model {self.current_model_type} loaded!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


    def draw_point_cloud(self, view, points, colors=None):
        try:
            # 清除原有显示
            for item in view.items[:]:
                if isinstance(item, gl.GLScatterPlotItem):
                    view.removeItem(item)

            if points is None or len(points) == 0:
                return

            # 颜色处理
            if colors is not None:
                # 转换为RGBA数组，alpha=1
                color_array = np.c_[colors[:, :3] / 255.0, np.ones((colors.shape[0], 1))]
            elif points.shape[1] >= 6:
                # 使用RGB通道，添加alpha
                color_array = np.c_[points[:, 3:6] / 255.0, np.ones((points.shape[0], 1))]
            else:
                # 使用高度映射到绿色通道
                z = points[:, 2]
                normalized_z = (z - z.min()) / (z.max() - z.min())
                color_array = np.zeros((points.shape[0], 4))
                color_array[:, 1] = normalized_z  # 绿色通道
                color_array[:, 3] = 1  # alpha=1

            # 创建点云对象
            plot_item = gl.GLScatterPlotItem(
                pos=points[:, :3],
                color=color_array,
                size=1,
                pxMode=True
            )
            view.addItem(plot_item)
            return plot_item
        except Exception as e:
            print(f"Error drawing point cloud: {e}")
            QMessageBox.warning(self, "Error", f"Failed to draw point cloud: {str(e)}")
            return None

    def load_point_cloud(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Point Cloud File", "", "Point Cloud Files (*.npy *.txt);;All Files (*)"
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

        # 特征补全
        if points.shape[1] < 9:
            points_with_features = np.zeros((points.shape[0], 9))
            points_with_features[:, :points.shape[1]] = points
            if points.shape[1] < 6:
                points_with_features[:, 3:6] = 255  # 默认白色
            points_with_features[:, 6:] = 0  # 默认法向量
            self.current_points = points_with_features
        else:
            self.current_points = points

        # 用原始文件名保存
        temp_dir = os.path.join(BASE_DIR, "temp_data")
        os.makedirs(temp_dir, exist_ok=True)
        # 获取原始文件名
        if hasattr(self.thread, 'file_path'):
            orig_name = os.path.basename(self.thread.file_path)
        else:
            orig_name = "temp.npy"
        self.temp_file = os.path.join(temp_dir, orig_name)
        np.save(self.temp_file, self.current_points)

        # 绘制原始点云
        self.current_plot = self.draw_point_cloud(self.original_view, self.current_points)

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

            # 模拟进度条更新
            self.progress_bar.setValue(20)

            accuracy = inference(self.model, self.temp_file, infer_res_save_path)

            # 更新进度条
            self.progress_bar.setValue(100)

            # 显示准确率
            self.accuracy_label.setText(f"Accuracy: {accuracy:.4f}")

            # 显示结果
            if os.path.exists(
                    infer_res_save_path + "/" + os.path.basename(self.temp_file).replace(".npy", "_pred.txt")):
                colored_points = np.loadtxt(
                    infer_res_save_path + "/" + os.path.basename(self.temp_file).replace(".npy", "_pred.txt"))
            else:
                colored_points = self.current_points
            self.draw_point_cloud(self.result_view, colored_points[:, :3], colored_points[:, 3:])

        except Exception as e:
            print(f"Error during inference: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed during inference: {str(e)}")

    def clear_views(self):
        # 清除所有显示项
        for view in [self.original_view, self.result_view]:
            for item in view.items[:]:
                if isinstance(item, (gl.GLScatterPlotItem, gl.GLGridItem)):
                    view.removeItem(item)
            '''
            # 重新添加网格
            grid = gl.GLGridItem()
            view.addItem(grid)
            '''
        # 重置进度条
        self.progress_bar.setValue(0)
        # 重置准确率标签
        self.accuracy_label.setText("Accuracy: N/A")

class _SyncGLViewWidget(gl.GLViewWidget):
    """同步视图控件（动态子类化实现）"""
    camera_updated = pyqtSignal(object)  # 新增同步信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_cam_params = None

    def _get_camera_params(self):
        """获取当前相机参数"""
        return {
            'distance': self.opts['distance'],
            'elevation': self.opts['elevation'],
            'azimuth': self.opts['azimuth']
        }

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        self._check_camera_change()

    def wheelEvent(self, ev):
        super().wheelEvent(ev)
        self._check_camera_change()

    def _check_camera_change(self):
        """检测相机参数变化并发射信号"""
        current_params = self._get_camera_params()
        if current_params != self._last_cam_params:
            self.camera_updated.emit(current_params)
            self._last_cam_params = current_params


def _enable_view_sync(viewer_instance):
    """动态为Viewer实例启用视图同步"""

    # 替换原始视图控件
    def replace_view(old_view):
        new_view = _SyncGLViewWidget()
        cam_pos = old_view.cameraPosition()
        cam_opts = {
            'distance': old_view.opts['distance'],
            'elevation': old_view.opts['elevation'],
            'azimuth': old_view.opts['azimuth']
        }
        new_view.setCameraPosition(**cam_opts)
        # 检查列表长度
        if len(old_view.items) > 1:
            new_view.addItem(old_view.items[1])  # 转移GridItem
        return new_view

    # 替换左侧视图
    new_original_view = replace_view(viewer_instance.original_view)
    viewer_instance.gl_layout.replaceWidget(
        viewer_instance.original_view, new_original_view)
    viewer_instance.original_view.deleteLater()
    viewer_instance.original_view = new_original_view

    # 替换右侧视图
    new_result_view = replace_view(viewer_instance.result_view)
    viewer_instance.gl_layout.replaceWidget(
        viewer_instance.result_view, new_result_view)
    viewer_instance.result_view.deleteLater()
    viewer_instance.result_view = new_result_view

    # 连接同步信号
    def sync_views(source_params, target_view):
        target_view.blockSignals(True)
        target_view.setCameraPosition(
            distance=source_params['distance'],
            elevation=source_params['elevation'],
            azimuth=source_params['azimuth']
        )
        target_view.blockSignals(False)

    new_original_view.camera_updated.connect(
        lambda p: sync_views(p, new_result_view))
    new_result_view.camera_updated.connect(
        lambda p: sync_views(p, new_original_view))


# 动态补丁原类初始化方法
_original_viewer_init = PointCloudViewer.__init__


def _new_viewer_init(self):
    _original_viewer_init(self)  # 执行原有初始化
    _enable_view_sync(self)  # 启用同步功能


PointCloudViewer.__init__ = _new_viewer_init

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
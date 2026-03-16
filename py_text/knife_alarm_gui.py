# 系统成品代码
# R3 (conf=0.70, hits=5, area=0, cooldown=3) —— 最均衡
# R2 (conf=0.70, hits=3, area=0, cooldown=1) —— 反应更快但更吵
# R4 (conf=0.80, hits=5, area=0, cooldown=3) —— 更保守，检出明显下降
# R1 (conf=0.70, hits=1, area=0, cooldown=0) —— 抖动/误报灾难级（564 次/分钟）
# R6 (conf=0.70, hits=5, area=0.01, cooldown=3) —— 面积门槛让检出掉太多
# R5 (conf=0.70, hits=3, area=0.01, cooldown=1) —— 同上，且更吵
# R7 (conf=0.85, hits=5, area=0.01, cooldown=3) —— 最稳但几乎不报（检出帧占比 0.0097，漏检风险极大）
# -*- coding: utf-8 -*-
"""
knife_alarm_gui.py
成品：YOLO(ultralytics) + 规则判定 + 只显示命中框 + 非阻塞通知(托盘气泡) + 提示音
支持：摄像头 / 视频文件 二选一
"""

import os
import sys
import time
from dataclasses import dataclass
from collections import deque

import cv2

# -----------------------------
# 路线A：你可以在这里手动写死路径（方式1）
# -----------------------------
DEFAULT_MODEL_PATH = r"E:\User\ultralytics-8.3.241\runs\detect\yolov8m_150_no_copy_paste\weights\best.pt"
DEFAULT_VIDEO_PATH = r"E:\User\ultralytics-8.3.241\video\video_1.mp4"

# Windows 提示音（非阻塞）
try:
    import winsound
    HAS_WINSOUND = True
except Exception:
    HAS_WINSOUND = False

# ultralytics
from ultralytics import YOLO

# PyQt5
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QRadioButton, QButtonGroup, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QMessageBox, QSystemTrayIcon
)


@dataclass
class RuleConfig:
    # 规则阈值
    conf_th: float = 0.70          # 置信度阈值（命中阈值）
    hits_required: int = 3         # 命中需要的 hit 数（连续/累计窗口内）
    hit_window_sec: float = 1.0    # 统计窗口（秒）
    min_area_ratio: float = 0.00   # 最小框面积比例（框面积/图像面积），过滤小噪点框
    cooldown_sec: float = 1.0      # 触发后冷却时间（秒），防止连发

    # 推理参数
    imgsz: int = 640
    iou: float = 0.5
    raw_conf: float = 0.001        # 模型输出的最低 conf（尽量低，避免你“看不到”潜在命中）
    device: str = "0"              # "0" GPU / "cpu"
    max_det: int = 300

    # 提示
    enable_sound: bool = True
    enable_notify: bool = True


class VideoWorker(QThread):
    frame_signal = pyqtSignal(QImage)          # 给界面显示
    status_signal = pyqtSignal(str)            # 状态栏
    notify_signal = pyqtSignal(str, str)       # (title, msg)
    stopped_signal = pyqtSignal()

    def __init__(self, model_path: str, source_mode: str, video_path: str, cam_index: int, rule: RuleConfig):
        super().__init__()
        self.model_path = model_path
        self.source_mode = source_mode  # "video" or "camera"
        self.video_path = video_path
        self.cam_index = cam_index
        self.rule = rule

        self._stop_flag = False

        # hits 统计：保存最近窗口内的 hit 时间戳
        self.hit_times = deque()
        self.last_trigger_time = 0.0

    def stop(self):
        self._stop_flag = True

    def _play_sound_non_block(self):
        if not self.rule.enable_sound:
            return
        if not HAS_WINSOUND:
            return
        # winsound.PlaySound(SND_ASYNC) 是异步不阻塞
        try:
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            pass

    def _within_cooldown(self, now: float) -> bool:
        return (now - self.last_trigger_time) < self.rule.cooldown_sec

    def _push_hit(self, now: float):
        # 记录一次 hit
        self.hit_times.append(now)
        # 清理窗口外的 hit
        window = self.rule.hit_window_sec
        while self.hit_times and (now - self.hit_times[0]) > window:
            self.hit_times.popleft()

    def _hit_count(self, now: float) -> int:
        window = self.rule.hit_window_sec
        while self.hit_times and (now - self.hit_times[0]) > window:
            self.hit_times.popleft()
        return len(self.hit_times)

    def run(self):
        # 1) 加载模型
        if not os.path.exists(self.model_path):
            self.status_signal.emit(f"模型不存在：{self.model_path}")
            self.stopped_signal.emit()
            return

        try:
            model = YOLO(self.model_path)
        except Exception as e:
            self.status_signal.emit(f"模型加载失败：{e}")
            self.stopped_signal.emit()
            return

        # 2) 打开视频源
        if self.source_mode == "video":
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.status_signal.emit(f"视频打不开：{self.video_path}")
                self.stopped_signal.emit()
                return
        else:
            cap = cv2.VideoCapture(self.cam_index)
            if not cap.isOpened():
                self.status_signal.emit(f"摄像头打不开：index={self.cam_index}")
                self.stopped_signal.emit()
                return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1e-3:
            fps = 30.0

        frame_id = 0
        t0 = time.time()

        while not self._stop_flag:
            ok, frame = cap.read()
            if not ok:
                break

            frame_id += 1
            h, w = frame.shape[:2]
            img_area = float(w * h)

            # 3) 推理（raw_conf 设低，之后用规则 conf_th 再过滤）
            t_infer0 = time.time()
            try:
                results = model.predict(
                    source=frame,
                    imgsz=self.rule.imgsz,
                    conf=self.rule.raw_conf,
                    iou=self.rule.iou,
                    device=self.rule.device,
                    verbose=False,
                    max_det=self.rule.max_det
                )
            except Exception as e:
                self.status_signal.emit(f"推理失败：{e}")
                break
            infer_ms = (time.time() - t_infer0) * 1000.0

            # 4) 从结果里找“本帧最可信的候选框”
            best = None  # (conf, xyxy, cls)
            raw_count = 0

            r0 = results[0]
            if r0.boxes is not None and len(r0.boxes) > 0:
                for b in r0.boxes:
                    raw_count += 1
                    conf = float(b.conf.item()) if b.conf is not None else 0.0
                    cls = int(b.cls.item()) if b.cls is not None else -1
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / img_area

                    # 规则过滤：conf + 面积
                    if conf < self.rule.conf_th:
                        continue
                    if area_ratio < self.rule.min_area_ratio:
                        continue

                    if best is None or conf > best[0]:
                        best = (conf, (x1, y1, x2, y2), cls, area_ratio)

            now = time.time()

            # 5) 规则判定：本帧是否 hit
            hit_this_frame = 1 if best is not None else 0
            if hit_this_frame:
                self._push_hit(now)

            hits = self._hit_count(now)

            # 6) 触发条件：hits 达标 + 不在 cooldown
            triggered = 0
            if hits >= self.rule.hits_required and (not self._within_cooldown(now)):
                triggered = 1
                self.last_trigger_time = now

                # 非阻塞提示：托盘气泡 + 音效
                if self.rule.enable_notify:
                    self.notify_signal.emit(
                        "刀具预警",
                        f"规则命中：conf≥{self.rule.conf_th:.2f}, hits={hits}/{self.rule.hits_required}, area≥{self.rule.min_area_ratio:.3f}"
                    )
                self._play_sound_non_block()

            # 7) 可视化：只画“命中框”
            vis = frame.copy()

            if best is not None:
                conf, (x1, y1, x2, y2), cls, area_ratio = best
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # 命中框（仅命中时画）
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"HIT conf={conf:.2f} area={area_ratio:.3f}"
                cv2.putText(vis, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 8) 叠加调试信息（帮助你判断：没检测到 vs 被规则过滤）
            # 你之前看到“一堆框”，那是 raw 全部输出；这里我们只显示命中框，但保留统计信息。
            elapsed = time.time() - t0
            t_sec = frame_id / fps
            info1 = f"frame {frame_id}  t={t_sec:.2f}s  fps={fps:.1f}  infer={infer_ms:.1f}ms"
            info2 = f"raw_boxes={raw_count}  hit={hit_this_frame}  hits={hits}/{self.rule.hits_required}  trig={triggered}  cd={max(0.0, self.rule.cooldown_sec-(now-self.last_trigger_time)):.1f}s"
            info3 = f"CONF_TH={self.rule.conf_th:.2f}  MIN_AREA={self.rule.min_area_ratio:.3f}  WIN={self.rule.hit_window_sec:.1f}s"
            y0 = 25
            for s in [info1, info2, info3]:
                cv2.putText(vis, s, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(vis, s, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)
                y0 += 25

            # 9) 发给 UI 显示
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
            self.frame_signal.emit(qimg.copy())
            self.status_signal.emit(f"运行中：frame={frame_id}, raw={raw_count}, hit={hit_this_frame}, hits={hits}, trig={triggered}")

            # 控制一下线程节奏（让 UI 更丝滑）
            self.msleep(1)

        cap.release()
        self.status_signal.emit("已停止")
        self.stopped_signal.emit()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("刀具检测预警系统")
        self.resize(1100, 700)

        self.worker = None
        self.rule = RuleConfig()

        # --- 托盘通知（Windows 风格，不阻塞，自动消失）
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(self.style().standardIcon(self.style().SP_MessageBoxWarning))
        self.tray.setVisible(True)

        # --- 画面显示
        self.video_label = QLabel("点击开始后显示画面")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 500)
        self.video_label.setStyleSheet("background:#111; color:#ddd; border:1px solid #333;")

        # --- 状态
        self.status = QLabel("就绪")
        self.status.setStyleSheet("color:#333;")

        # --- 模型路径（方式2：界面选择）
        self.model_path_label = QLabel(DEFAULT_MODEL_PATH)
        self.btn_pick_model = QPushButton("选择模型best.pt")
        self.btn_pick_model.clicked.connect(self.pick_model)

        # --- 输入源选择
        self.rb_video = QRadioButton("视频文件")
        self.rb_cam = QRadioButton("摄像头")
        self.rb_video.setChecked(True)
        self.source_group = QButtonGroup()
        self.source_group.addButton(self.rb_video)
        self.source_group.addButton(self.rb_cam)

        self.video_path_label = QLabel(DEFAULT_VIDEO_PATH)
        self.btn_pick_video = QPushButton("选择视频")
        self.btn_pick_video.clicked.connect(self.pick_video)

        self.cam_index = QSpinBox()
        self.cam_index.setRange(0, 10)
        self.cam_index.setValue(0)

        # --- 规则参数
        self.conf_th = QDoubleSpinBox()
        self.conf_th.setRange(0.0, 1.0)
        self.conf_th.setSingleStep(0.01)
        self.conf_th.setValue(self.rule.conf_th)

        self.hits_required = QSpinBox()
        self.hits_required.setRange(1, 30)
        self.hits_required.setValue(self.rule.hits_required)

        self.hit_window_sec = QDoubleSpinBox()
        self.hit_window_sec.setRange(0.1, 10.0)
        self.hit_window_sec.setSingleStep(0.1)
        self.hit_window_sec.setValue(self.rule.hit_window_sec)

        self.min_area_ratio = QDoubleSpinBox()
        self.min_area_ratio.setRange(0.0, 1.0)
        self.min_area_ratio.setSingleStep(0.001)
        self.min_area_ratio.setDecimals(3)
        self.min_area_ratio.setValue(self.rule.min_area_ratio)

        self.cooldown_sec = QDoubleSpinBox()
        self.cooldown_sec.setRange(0.0, 10.0)
        self.cooldown_sec.setSingleStep(0.1)
        self.cooldown_sec.setValue(self.rule.cooldown_sec)

        # --- 推理参数
        self.imgsz = QSpinBox()
        self.imgsz.setRange(320, 1280)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(self.rule.imgsz)

        self.iou = QDoubleSpinBox()
        self.iou.setRange(0.1, 0.95)
        self.iou.setSingleStep(0.05)
        self.iou.setValue(self.rule.iou)

        self.device_label = QLabel("[device: 0(GPU) 或 cpu]:")
        self.device_value = QLabel(self.rule.device)

        # --- 控制按钮
        self.btn_start = QPushButton("开始")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)

        # --- 布局
        left = QVBoxLayout()
        left.addWidget(self.video_label)
        left.addWidget(self.status)

        cfg_box = QGroupBox("配置")
        form = QFormLayout()

        form.addRow(QLabel("模型路径："), self.model_path_label)
        form.addRow(self.btn_pick_model)

        form.addRow(QLabel("输入源："), self._hbox(self.rb_video, self.rb_cam))
        form.addRow(QLabel("视频："), self.video_path_label)
        form.addRow(self.btn_pick_video)
        form.addRow(QLabel("摄像头index："), self.cam_index)

        form.addRow(QLabel("conf_th："), self.conf_th)
        form.addRow(QLabel("hits_required："), self.hits_required)
        form.addRow(QLabel("hit_window_sec："), self.hit_window_sec)
        form.addRow(QLabel("min_area_ratio："), self.min_area_ratio)
        form.addRow(QLabel("cooldown_sec："), self.cooldown_sec)

        form.addRow(QLabel("imgsz："), self.imgsz)
        form.addRow(QLabel("iou："), self.iou)
        form.addRow(self.device_label, self.device_value)

        form.addRow(self._hbox(self.btn_start, self.btn_stop))

        cfg_box.setLayout(form)

        right = QVBoxLayout()
        right.addWidget(cfg_box)
        right.addStretch(1)

        root = QHBoxLayout()
        root.addLayout(left, 3)
        root.addLayout(right, 1)
        self.setLayout(root)

    def _hbox(self, *widgets):
        box = QHBoxLayout()
        w = QWidget()
        for it in widgets:
            box.addWidget(it)
        w.setLayout(box)
        return w

    def pick_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", os.path.dirname(DEFAULT_MODEL_PATH), "PyTorch (*.pt)")
        if path:
            self.model_path_label.setText(path)

    def pick_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", os.path.dirname(DEFAULT_VIDEO_PATH), "Video (*.mp4 *.avi *.mkv *.mov)")
        if path:
            self.video_path_label.setText(path)

    def start(self):
        if self.worker is not None:
            return

        # 读取 UI 参数 -> rule
        self.rule.conf_th = float(self.conf_th.value())
        self.rule.hits_required = int(self.hits_required.value())
        self.rule.hit_window_sec = float(self.hit_window_sec.value())
        self.rule.min_area_ratio = float(self.min_area_ratio.value())
        self.rule.cooldown_sec = float(self.cooldown_sec.value())
        self.rule.imgsz = int(self.imgsz.value())
        self.rule.iou = float(self.iou.value())

        model_path = self.model_path_label.text().strip()
        if not model_path:
            QMessageBox.warning(self, "提示", "请先选择模型")
            return

        source_mode = "video" if self.rb_video.isChecked() else "camera"
        video_path = self.video_path_label.text().strip()
        cam_index = int(self.cam_index.value())

        if source_mode == "video" and (not video_path or not os.path.exists(video_path)):
            QMessageBox.warning(self, "提示", "视频路径无效，请选择视频文件")
            return

        self.worker = VideoWorker(model_path, source_mode, video_path, cam_index, self.rule)
        self.worker.frame_signal.connect(self.on_frame)
        self.worker.status_signal.connect(self.on_status)
        self.worker.notify_signal.connect(self.on_notify)
        self.worker.stopped_signal.connect(self.on_stopped)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status.setText("启动中...")
        self.worker.start()

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.btn_stop.setEnabled(False)

    def on_frame(self, qimg: QImage):
        # 自适应缩放
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_status(self, s: str):
        self.status.setText(s)

    def on_notify(self, title: str, msg: str):
        # Windows 托盘通知（不阻塞、自动消失）
        try:
            self.tray.showMessage(title, msg, QSystemTrayIcon.Warning, 2000)
        except Exception:
            pass

    def on_stopped(self):
        self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon())  # 你也可以换成自己的 ico
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

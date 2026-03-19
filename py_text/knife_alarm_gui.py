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


# 默认路径

DEFAULT_MODEL_PATH = r"E:\User\ultralytics-8.3.241\runs\detect\yolov8m_150_no_copy_paste\weights\best.pt"
DEFAULT_VIDEO_PATH = r"E:\User\ultralytics-8.3.241\video\video_1.mp4"
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# Windows 提示音
try:
    import winsound
    HAS_WINSOUND = True
except Exception:
    HAS_WINSOUND = False

try:
    import torch
    HAS_CUDA = bool(torch.cuda.is_available())
except Exception:
    HAS_CUDA = False

# ultralytics
from ultralytics import YOLO

# PyQt5
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QRectF, QPointF, QLineF
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QColor, QPen, QBrush, QPainterPath, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QRadioButton, QButtonGroup, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QMessageBox, QSystemTrayIcon, QComboBox, QGridLayout
)
# 日志模块
from log_system import LogManager, LogViewerDialog


# 趋势图控件：用于绘制最近一段时间内的数值变化，如置信度和推理耗时。
class TrendChartWidget(QWidget):
    # 初始化图表范围、阈值和缓存队列。
    def __init__(self, max_points=60, y_min=0.0, y_max=1.0, threshold=None, value_suffix="", value_decimals=2, parent=None):
        super().__init__(parent)
        self.max_points = max_points
        self.y_min = y_min
        self.y_max = y_max
        self.threshold = threshold
        self.value_suffix = value_suffix
        self.value_decimals = value_decimals
        self.values = deque(maxlen=max_points)
        self.setMinimumHeight(110)

    # 清空当前曲线缓存，并立即刷新界面。
    def clear(self):
        self.values.clear()
        self.update()

    # 更新阈值线位置，供界面实时显示当前规则门槛。
    def set_threshold(self, value):
        self.threshold = value
        self.update()

    # 追加一个新采样点，超出容量后自动丢弃最旧数据。
    def append_value(self, value):
        try:
            v = float(value)
        except Exception:
            v = 0.0
        self.values.append(v)
        self.update()

    # 计算纵轴上限；若未固定上限，则根据当前数据自适应。
    def _effective_ymax(self):
        if self.y_max is not None:
            return max(self.y_max, self.y_min + 1e-6)
        if not self.values:
            return 1.0
        current_max = max(self.values)
        if current_max <= self.y_min:
            return self.y_min + 1.0
        return current_max * 1.25

    # 自定义绘制曲线、网格、阈值线和当前值文本。
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(8, 8, -8, -8)
        painter.fillRect(rect, QColor("#fcfcfc"))
        painter.setPen(QPen(QColor("#d9d9d9"), 1))
        painter.drawRect(rect)

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        left = rect.left() + 46
        top = rect.top() + 18
        right = rect.right() - 12
        bottom = rect.bottom() - 22
        plot_rect = QRectF(left, top, max(10, right - left), max(10, bottom - top))

        grid_pen = QPen(QColor("#e8e8e8"), 1)
        painter.setPen(grid_pen)
        for i in range(6):
            y = plot_rect.top() + i * plot_rect.height() / 5.0
            painter.drawLine(QLineF(plot_rect.left(), y, plot_rect.right(), y))
        for i in range(7):
            x = plot_rect.left() + i * plot_rect.width() / 6.0
            painter.drawLine(QLineF(x, plot_rect.top(), x, plot_rect.bottom()))

        y_min = self.y_min
        y_max = self._effective_ymax()

        painter.setPen(QPen(QColor("#666666"), 1))
        top_text = f"{y_max:.1f}" if y_max > 1 else f"{y_max:.2f}"
        bottom_text = f"{y_min:.1f}" if y_max > 1 else f"{y_min:.2f}"
        painter.drawText(QRectF(plot_rect.left() - 42, plot_rect.top() - 8, 36, 16), Qt.AlignRight | Qt.AlignVCenter, top_text)
        painter.drawText(QRectF(plot_rect.left() - 42, plot_rect.bottom() - 8, 36, 16), Qt.AlignRight | Qt.AlignVCenter, bottom_text)

        if self.threshold is not None:
            threshold_value = max(y_min, min(self.threshold, y_max))
            y = plot_rect.bottom() - (threshold_value - y_min) / (y_max - y_min) * plot_rect.height()
            threshold_pen = QPen(QColor("#d9534f"), 1)
            threshold_pen.setStyle(Qt.DashLine)
            painter.setPen(threshold_pen)
            painter.drawLine(QLineF(plot_rect.left(), y, plot_rect.right(), y))

        if len(self.values) >= 2:
            points = []
            count = len(self.values)
            for i, value in enumerate(self.values):
                x = plot_rect.left() + i * plot_rect.width() / max(1, count - 1)
                value = max(y_min, min(value, y_max))
                y = plot_rect.bottom() - (value - y_min) / (y_max - y_min) * plot_rect.height()
                points.append(QPointF(x, y))

            fill_path = QPainterPath()
            fill_path.moveTo(points[0].x(), plot_rect.bottom())
            for p in points:
                fill_path.lineTo(p)
            fill_path.lineTo(points[-1].x(), plot_rect.bottom())
            fill_path.closeSubpath()
            painter.fillPath(fill_path, QBrush(QColor(102, 178, 255, 60)))

            line_path = QPainterPath()
            line_path.moveTo(points[0])
            for p in points[1:]:
                line_path.lineTo(p)
            painter.setPen(QPen(QColor("#4a90e2"), 2))
            painter.drawPath(line_path)

        elif len(self.values) == 1:
            value = max(y_min, min(self.values[0], y_max))
            y = plot_rect.bottom() - (value - y_min) / (y_max - y_min) * plot_rect.height()
            painter.setPen(QPen(QColor("#4a90e2"), 2))
            painter.drawLine(QLineF(plot_rect.left(), y, plot_rect.right(), y))

        current_value = self.values[-1] if self.values else 0.0
        painter.setPen(QPen(QColor("#222222"), 1))
        if self.value_suffix:
            text = f"{current_value:.{self.value_decimals}f} {self.value_suffix}"
        else:
            text = f"{current_value:.{self.value_decimals}f}"
        painter.drawText(QRectF(plot_rect.left(), rect.top(), plot_rect.width(), 16), Qt.AlignRight | Qt.AlignVCenter, text)


# 命中进度控件：分别显示“显示命中”和“报警命中”的累计进度。
class HitsProgressWidget(QWidget):
    # 初始化当前命中数与两个目标阈值。
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_hits = 0
        self.display_required = 0
        self.alert_required = 0
        self.setMinimumHeight(100)

    # 复位进度条状态。
    def clear(self):
        self.current_hits = 0
        self.update()

    # 更新当前命中值和两个阶段的目标值。
    def set_values(self, current_hits, display_required, alert_required):
        self.current_hits = int(current_hits)
        self.display_required = int(display_required)
        self.alert_required = int(alert_required)
        self.update()

    # 绘制单条进度条及其右侧数值。
    def _draw_progress(self, painter, rect, label, current, target, color):
        painter.setPen(QPen(QColor("#333333"), 1))
        painter.drawText(QRectF(rect.left(), rect.top() - 18, rect.width(), 16), Qt.AlignLeft | Qt.AlignVCenter, label)
        painter.setPen(QPen(QColor("#d9d9d9"), 1))
        painter.setBrush(QBrush(QColor("#f2f2f2")))
        painter.drawRoundedRect(rect, 6, 6)

        ratio = 0.0
        if target > 0:
            ratio = max(0.0, min(float(current) / float(target), 1.0))

        fill_rect = QRectF(rect.left(), rect.top(), rect.width() * ratio, rect.height())
        painter.setBrush(QBrush(QColor(color)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(fill_rect, 6, 6)

        painter.setPen(QPen(QColor("#222222"), 1))
        painter.drawText(QRectF(rect.left(), rect.top(), rect.width() - 8, rect.height()), Qt.AlignRight | Qt.AlignVCenter, f"{current} / {target}")

    # 绘制两个阶段对应的命中进度条。
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        rect = self.rect().adjusted(8, 8, -8, -8)
        painter.fillRect(rect, QColor("#fcfcfc"))
        painter.setPen(QPen(QColor("#d9d9d9"), 1))
        painter.drawRect(rect)

        bar_width = rect.width() - 24
        first_rect = QRectF(rect.left() + 12, rect.top() + 22, bar_width, 18)
        second_rect = QRectF(rect.left() + 12, rect.top() + 58, bar_width, 18)

        self._draw_progress(
            painter,
            first_rect,
            "显示命中进度",
            self.current_hits,
            self.display_required,
            "#f0ad4e"
        )
        self._draw_progress(
            painter,
            second_rect,
            "报警命中进度",
            self.current_hits,
            self.alert_required,
            "#d9534f"
        )


# 阶段指示控件：展示“无/候选/确认/报警”四个运行阶段。
class StageIndicatorWidget(QWidget):
    # 初始化阶段名称与报警次数显示。
    def __init__(self, parent=None):
        super().__init__(parent)
        self.stage = "无"
        self.triggered = 0
        self.stages = ["无", "候选", "确认", "报警"]
        self.setMinimumHeight(110)

    # 复位阶段显示。
    def clear(self):
        self.stage = "无"
        self.triggered = 0
        self.update()

    # 更新当前阶段与累计报警次数。
    def set_stage_state(self, stage, triggered):
        self.stage = str(stage)
        self.triggered = int(triggered)
        self.update()

    # 按阶段绘制状态圆点、连线与文字信息。
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        rect = self.rect().adjusted(8, 8, -8, -8)
        painter.fillRect(rect, QColor("#fcfcfc"))
        painter.setPen(QPen(QColor("#d9d9d9"), 1))
        painter.drawRect(rect)

        title_rect = QRectF(rect.left() + 12, rect.top() + 4, rect.width() - 24, 16)
        painter.setPen(QPen(QColor("#222222"), 1))
        painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, f"当前阶段：{self.stage}")

        center_y = rect.center().y() + 4
        left = rect.left() + 38
        right = rect.right() - 38
        step = (right - left) / max(1, len(self.stages) - 1)

        active_colors = {
            "无": QColor("#9e9e9e"),
            "候选": QColor("#f0ad4e"),
            "确认": QColor("#ff8c42"),
            "报警": QColor("#d9534f"),
        }

        painter.setPen(QPen(QColor("#cfcfcf"), 2))
        if len(self.stages) > 1:
            painter.drawLine(QLineF(left, center_y, right, center_y))

        for i, stage_name in enumerate(self.stages):
            x = left + step * i
            is_active = (stage_name == self.stage)
            color = active_colors.get(stage_name, QColor("#9e9e9e")) if is_active else QColor("#d6d6d6")

            painter.setPen(QPen(QColor("#bcbcbc"), 1))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QPointF(x, center_y), 8, 8)

            painter.setPen(QPen(QColor("#333333"), 1))
            painter.drawText(QRectF(x - 24, center_y + 12, 48, 16), Qt.AlignCenter, stage_name)

        painter.setPen(QPen(QColor("#222222"), 1))
        painter.drawText(QRectF(rect.left() + 12, rect.bottom() - 20, rect.width() - 24, 14), Qt.AlignLeft | Qt.AlignVCenter, f"报警触发：{self.triggered}")


@dataclass
# 规则配置数据类：统一保存判定规则、推理参数和提示选项。
class RuleConfig:
    # 规则阈值
    conf_th: float = 0.70          # 置信度阈值
    hits_required: int = 15        # 命中需要的 hit 数
    hit_window_sec: float = 1.0    # 统计窗口（秒）
    min_area_ratio: float = 0.00   # 最小框面积比例（框面积/图像面积）
    cooldown_sec: float = 1.0      # 触发后冷却时间（秒），防止连发

    # 推理参数
    imgsz: int = 640
    iou: float = 0.5
    raw_conf: float = 0.001        # 模型输出的最低 conf
    device: str = "0"              # "0" GPU / "cpu"
    max_det: int = 300

    # 分级显示
    display_hits_required: int = 10 # 正式显示命中框所需的 hit 数

    # 提示
    enable_sound: bool = True
    enable_notify: bool = True


# 工作线程：负责视频读取、模型推理、规则判定和信号回传，避免阻塞主界面。
class VideoWorker(QThread):
    frame_signal = pyqtSignal(QImage)          # 给界面显示
    status_signal = pyqtSignal(str)            # 状态栏
    notify_signal = pyqtSignal(str, str)       # (title, msg)
    info_signal = pyqtSignal(dict)             # 运行信息
    stopped_signal = pyqtSignal()
    finished_signal = pyqtSignal()
    system_log_signal = pyqtSignal(str, str)
    event_log_signal = pyqtSignal(dict)

    # 保存模型路径、输入源信息和规则配置，并初始化运行期统计变量。
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
        self.prev_stage = "无"
        self.total_hits = 0
        self.total_triggers = 0
        self.infer_sum_ms = 0.0
        self.infer_max_ms = 0.0
        self.run_start_time = ""

    # 对外提供停止标记，供主线程安全结束检测循环。
    def stop(self):
        self._stop_flag = True

    # 异步播放系统提示音，避免报警时阻塞检测线程。
    def _play_sound_non_block(self):
        if not self.rule.enable_sound:
            return
        if not HAS_WINSOUND:
            return
        try:
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            pass

    # 判断当前时刻是否仍处于冷却期内。
    def _within_cooldown(self, now: float) -> bool:
        return (now - self.last_trigger_time) < self.rule.cooldown_sec

    # 写入一次命中时间，并移除统计窗口之外的旧命中。
    def _push_hit(self, now: float):
        # 记录一次 hit
        self.hit_times.append(now)
        # 清理窗口外的 hit
        window = self.rule.hit_window_sec
        while self.hit_times and (now - self.hit_times[0]) > window:
            self.hit_times.popleft()

    # 返回当前统计窗口内的有效命中次数。
    def _hit_count(self, now: float) -> int:
        window = self.rule.hit_window_sec
        while self.hit_times and (now - self.hit_times[0]) > window:
            self.hit_times.popleft()
        return len(self.hit_times)

    # 线程主流程：加载模型、打开输入源、逐帧推理、规则判定、可视化并发送结果到界面。
    def run(self):
        self.run_start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        # 加载模型
        if not os.path.exists(self.model_path):
            self.status_signal.emit(f"模型不存在：{self.model_path}")
            self.system_log_signal.emit("ERROR", f"模型不存在：{self.model_path}")
            self.stopped_signal.emit()
            self.finished_signal.emit()
            return

        try:
            model = YOLO(self.model_path)
            self.system_log_signal.emit("INFO", f"模型加载成功：{self.model_path}")
        except Exception as e:
            self.status_signal.emit(f"模型加载失败：{e}")
            self.system_log_signal.emit("ERROR", f"模型加载失败：{e}")
            self.stopped_signal.emit()
            self.finished_signal.emit()
            return
                # 打开视频源
        if self.source_mode == "video":
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.status_signal.emit(f"视频打不开：{self.video_path}")
                self.system_log_signal.emit("ERROR", f"视频打不开：{self.video_path}")
                self.stopped_signal.emit()
                self.finished_signal.emit()
                return
            self.system_log_signal.emit("INFO", f"视频打开成功：{self.video_path}")
        else:
            cap = cv2.VideoCapture(self.cam_index)
            if not cap.isOpened():
                self.status_signal.emit(f"摄像头打不开：index={self.cam_index}")
                self.system_log_signal.emit("ERROR", f"摄像头打不开：index={self.cam_index}")
                self.stopped_signal.emit()
                self.finished_signal.emit()
                return
            self.system_log_signal.emit("INFO", f"摄像头打开成功：index={self.cam_index}")

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

            # 推理（raw_conf 设低，之后用规则 conf_th 再过滤）
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
                self.system_log_signal.emit("ERROR", f"推理失败：{e}")
                break
            infer_ms = (time.time() - t_infer0) * 1000.0
            self.infer_sum_ms += infer_ms
            if infer_ms > self.infer_max_ms:
                self.infer_max_ms = infer_ms

            # 收集“本帧所有有效框”，同时保留一个最优框用于统计显示
            valid_boxes = []   # [(conf, (x1, y1, x2, y2), cls, area_ratio), ...]
            best = None
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

                    item = (conf, (x1, y1, x2, y2), cls, area_ratio)
                    valid_boxes.append(item)

                    if best is None or conf > best[0]:
                        best = item

            now = time.time()

           # 规则判定：这一帧只要存在至少一个有效框，就记 1 次 hit
            hit_this_frame = 1 if len(valid_boxes) > 0 else 0
            if hit_this_frame:
                self._push_hit(now)
                self.total_hits += 1

            hits = self._hit_count(now)

            # 触发条件：hits 达标 + 不在 cooldown
            triggered = 0
            if hits >= self.rule.hits_required and (not self._within_cooldown(now)):
                triggered = 1
                self.last_trigger_time = now
                self.total_triggers += 1

                # 非阻塞提示：托盘气泡 + 音效
                if self.rule.enable_notify:
                    self.notify_signal.emit(
                        "刀具预警",
                        f"规则命中：conf≥{self.rule.conf_th:.2f}，命中次数={hits}/{self.rule.hits_required}，"
                        "面积≥{self.rule.min_area_ratio:.3f}"
                    )
                self._play_sound_non_block()

            display_candidate = len(valid_boxes) > 0
            display_confirmed = display_candidate and (hits >= self.rule.display_hits_required)
            display_alert = display_confirmed and (triggered == 1 or self._within_cooldown(now))

            # 可视化：画“所有有效框”，但报警仍按整帧单次判定
            vis = frame.copy()

            for item in valid_boxes:
                conf, (x1, y1, x2, y2), cls, area_ratio = item
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                if display_alert:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
                elif display_confirmed:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                elif display_candidate:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)

            # 叠加调试信息
            elapsed = time.time() - t0
            t_sec = frame_id / fps
            if display_alert:
                stage = "报警"
            elif display_confirmed:
                stage = "确认"
            elif display_candidate:
                stage = "候选"
            else:
                stage = "无"

            best_conf = 0.0
            best_area_ratio = 0.0
            if best is not None:
                best_conf = best[0]
                best_area_ratio = best[3]

            cooldown_left = max(0.0, self.rule.cooldown_sec - (now - self.last_trigger_time))

            if stage != self.prev_stage:
                self.system_log_signal.emit("INFO", f"阶段变化：frame={frame_id}，{self.prev_stage} -> {stage}")
                self.event_log_signal.emit({
                    "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "event_type": f"stage_{stage}",
                    "frame_id": frame_id,
                    "stage": stage,
                    "best_conf": f"{best_conf:.4f}",
                    "hits": hits,
                    "hits_required": self.rule.hits_required,
                    "source_mode": self.source_mode,
                    "video_path": self.video_path,
                    "cam_index": self.cam_index
                })
                self.prev_stage = stage

            if triggered == 1:
                self.system_log_signal.emit("WARNING", f"报警触发：frame={frame_id}，conf={best_conf:.4f}，hits={hits}/{self.rule.hits_required}")
                self.event_log_signal.emit({
                    "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "event_type": "triggered",
                    "frame_id": frame_id,
                    "stage": stage,
                    "best_conf": f"{best_conf:.4f}",
                    "hits": hits,
                    "hits_required": self.rule.hits_required,
                    "source_mode": self.source_mode,
                    "video_path": self.video_path,
                    "cam_index": self.cam_index
                })

            self.info_signal.emit({
                "stage": stage,
                "frame_id": frame_id,
                "fps": fps,
                "infer_ms": infer_ms,
                "raw_count": raw_count,
                "valid_count": len(valid_boxes),
                "hit_this_frame": hit_this_frame,
                "hits": hits,
                "hits_required": self.rule.hits_required,
                "triggered": triggered,
                "best_conf": best_conf,
            })

            # 发给 UI 显示
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
            self.frame_signal.emit(qimg.copy())
            self.status_signal.emit("运行中")

            # 控制一下线程节奏
            self.msleep(1)

        cap.release()
        self.status_signal.emit("已停止")
        self.system_log_signal.emit("INFO", f"运行结束：总帧数={frame_id}，总命中={self.total_hits}，总报警={self.total_triggers}")
        self.finished_signal.emit()
        self.stopped_signal.emit()
        self.event_log_signal.emit({
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": "run_finished",
            "frame_id": frame_id,
            "stage": self.prev_stage,
            "best_conf": "",
            "hits": self.total_hits,
            "hits_required": self.rule.hits_required,
            "source_mode": self.source_mode,
            "video_path": self.video_path,
            "cam_index": self.cam_index
        })


# 主窗口：负责界面搭建、参数收集、线程控制以及日志查看。
class MainWindow(QWidget):
    # 初始化规则、日志模块、托盘和所有界面子模块。
    def __init__(self):
        super().__init__()
        self.setWindowTitle("刀具检测预警系统")
        self.resize(1180, 760)

        self.worker = None
        self.rule = RuleConfig()
        self.camera_scan_range = 6
        self.available_cameras = {}
        self.log_dir = DEFAULT_LOG_DIR
        self.log_manager = LogManager(self.log_dir)
        self.log_viewer = LogViewerDialog(self)
        self.log_manager.set_viewer(self.log_viewer)
        self.run_start_time = ""
        self.run_stats = {
            "total_frames": 0,
            "infer_sum_ms": 0.0,
            "infer_max_ms": 0.0,
            "total_hits": 0,
            "total_triggers": 0
        }

        # 初始化界面模块
        self._init_tray()
        self._init_video_area()
        self._init_status_area()
        self._init_path_widgets()
        self._init_source_widgets()
        self._init_rule_widgets()
        self._init_runtime_widgets()
        self._init_control_widgets()
        self._set_compact_widget_widths()
        self._build_layout()
        self._scan_cameras_on_startup()

        self.log_manager.info("程序启动完成")

    # 托盘通知
    def _init_tray(self):
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(self.style().standardIcon(self.style().SP_MessageBoxWarning))
        self.tray.setVisible(True)


    # 视频显示区域
    def _init_video_area(self):
        self.video_label = QLabel("点击开始后显示画面")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(720, 440)
        self.video_label.setStyleSheet("background:#111; color:#ddd; border:1px solid #333;")


    # 状态栏
    def _init_status_area(self):
        self.status = QLabel("就绪")
        self.status.setStyleSheet("color:#333;")


    # 路径选择
    def _init_path_widgets(self):
        # 模型路径
        self.model_path_label = QLabel(DEFAULT_MODEL_PATH)
        self.btn_pick_model = QPushButton("选择模型文件 best.pt")
        self.btn_pick_model.clicked.connect(self.pick_model)

        self.video_path_label = QLabel(DEFAULT_VIDEO_PATH)
        self.btn_pick_video = QPushButton("选择视频文件")
        self.btn_pick_video.clicked.connect(self.pick_video)

        self.log_dir_label = QLabel(self.log_dir)
        self.btn_pick_log_dir = QPushButton("设置日志目录")
        self.btn_pick_log_dir.clicked.connect(self.pick_log_dir)


    # 输入源选择

    def _init_source_widgets(self):
        self.rb_video = QRadioButton("视频文件")
        self.rb_cam = QRadioButton("摄像头")
        self.rb_video.setChecked(True)
        self.source_group = QButtonGroup()
        self.source_group.addButton(self.rb_video)
        self.source_group.addButton(self.rb_cam)

        self.cam_select = QComboBox()

        self.btn_scan_camera = QPushButton("重新扫描摄像头")
        self.btn_scan_camera.clicked.connect(self._scan_cameras_on_startup)


    # 规则参数

    def _init_rule_widgets(self):
        self.conf_th = QDoubleSpinBox()
        self.conf_th.setRange(0.0, 1.0)
        self.conf_th.setSingleStep(0.01)
        self.conf_th.setValue(self.rule.conf_th)

        self.hits_required = QSpinBox()
        self.hits_required.setRange(1, 30)
        self.hits_required.setValue(self.rule.hits_required)

        self.display_hits_required = QSpinBox()
        self.display_hits_required.setRange(1, 30)
        self.display_hits_required.setValue(self.rule.display_hits_required)

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

        self.imgsz = QSpinBox()
        self.imgsz.setRange(320, 1280)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(self.rule.imgsz)

        self.iou = QDoubleSpinBox()
        self.iou.setRange(0.1, 0.95)
        self.iou.setSingleStep(0.05)
        self.iou.setValue(self.rule.iou)

        self.device_label = QLabel("推理设备：")
        self.device_select = QComboBox()
        self.device_select.addItem("GPU (device=0)", "0")
        self.device_select.addItem("CPU", "cpu")
        if HAS_CUDA:
            self.device_select.setCurrentIndex(0)
        else:
            self.device_select.setCurrentIndex(1)


    # 运行信息

    def _init_runtime_widgets(self):
        self.info_frame = QLabel("0")
        self.info_fps = QLabel("0.0")
        self.info_raw = QLabel("0")
        self.info_hit = QLabel("0")

        for lb in [self.info_frame, self.info_fps, self.info_raw, self.info_hit]:
            lb.setStyleSheet("font-size: 10pt;")

        self.conf_chart = TrendChartWidget(max_points=60, y_min=0.0, y_max=1.0, threshold=self.rule.conf_th, value_suffix="", value_decimals=2)
        self.infer_chart = TrendChartWidget(max_points=60, y_min=0.0, y_max=None, threshold=None, value_suffix="ms", value_decimals=1)
        self.hits_progress = HitsProgressWidget()
        self.stage_indicator = StageIndicatorWidget()


    # 控制按钮

    def _init_control_widgets(self):
        self.btn_start = QPushButton("开始运行")
        self.btn_stop = QPushButton("停止运行")
        self.btn_view_log = QPushButton("查看日志")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_view_log.clicked.connect(self.show_log_viewer)


    # 控件宽度

    def _set_compact_widget_widths(self):
        self.btn_pick_video.setFixedWidth(150)
        self.btn_scan_camera.setFixedWidth(150)
        self.btn_pick_model.setFixedWidth(220)
        self.btn_pick_log_dir.setFixedWidth(150)
        self.btn_view_log.setFixedWidth(120)

        self.cam_select.setFixedWidth(220)
        self.device_select.setFixedWidth(220)

        for w in [
            self.conf_th, self.hits_required, self.display_hits_required,
            self.hit_window_sec, self.min_area_ratio, self.cooldown_sec,
            self.imgsz, self.iou
        ]:
            w.setFixedWidth(220)


    # 整体布局

    def _build_layout(self):
        left = QVBoxLayout()
        left.addWidget(self.video_label)
        left.addWidget(self.status)

        cfg_box = QGroupBox("参数配置")

        top_form = QFormLayout()
        top_form.addRow(QLabel("模型路径："), self.model_path_label)
        top_form.addRow(self.btn_pick_model)
        top_form.addRow(QLabel("日志目录："), self.log_dir_label)
        top_form.addRow(self.btn_pick_log_dir)
        top_form.addRow(QLabel("输入源："), self._hbox(self.rb_video, self.rb_cam))
        top_form.addRow(QLabel("视频路径："), self.video_path_label)
        top_form.addRow(QLabel("操作："), self._hbox(self.btn_pick_video, self.btn_scan_camera))

        param_form = QFormLayout()
        param_form.addRow(QLabel("摄像头选择："), self.cam_select)
        param_form.addRow(QLabel("置信度阈值："), self.conf_th)
        param_form.addRow(QLabel("报警命中次数："), self.hits_required)
        param_form.addRow(QLabel("显示命中次数："), self.display_hits_required)
        param_form.addRow(QLabel("统计窗口："), self.hit_window_sec)
        param_form.addRow(QLabel("最小面积占比："), self.min_area_ratio)
        param_form.addRow(QLabel("冷却时间："), self.cooldown_sec)
        param_form.addRow(QLabel("输入尺寸："), self.imgsz)
        param_form.addRow(QLabel("NMS IoU 阈值："), self.iou)
        param_form.addRow(self.device_label, self.device_select)

        param_widget = QWidget()
        param_widget.setLayout(param_form)

        info_box = QGroupBox("运行信息")
        info_grid = QGridLayout()
        info_grid.setContentsMargins(12, 10, 12, 10)
        info_grid.setHorizontalSpacing(14)
        info_grid.setVerticalSpacing(6)

        info_grid.addWidget(QLabel("当前帧："), 0, 0)
        info_grid.addWidget(self.info_frame, 0, 1)
        info_grid.addWidget(QLabel("FPS："), 0, 2)
        info_grid.addWidget(self.info_fps, 0, 3)

        info_grid.addWidget(QLabel("原始框数："), 1, 0)
        info_grid.addWidget(self.info_raw, 1, 1)
        info_grid.addWidget(QLabel("有效框数："), 1, 2)
        info_grid.addWidget(self.info_hit, 1, 3)

        info_box.setLayout(info_grid)

        hits_box = QGroupBox("累计命中可视化")
        hits_layout = QVBoxLayout()
        hits_layout.setContentsMargins(6, 6, 6, 6)
        hits_layout.addWidget(self.hits_progress)
        hits_box.setLayout(hits_layout)

        right_top = QVBoxLayout()
        right_top.addWidget(info_box)
        right_top.addWidget(hits_box)

        middle = QHBoxLayout()
        middle.addWidget(param_widget, 1)
        middle.addLayout(right_top, 1)

        visual_box = QGroupBox("运行可视化")
        visual_grid = QGridLayout()
        visual_grid.setHorizontalSpacing(8)
        visual_grid.setVerticalSpacing(8)

        conf_panel = QGroupBox("当前置信度折线图")
        conf_layout = QVBoxLayout()
        conf_layout.setContentsMargins(6, 6, 6, 6)
        conf_layout.addWidget(self.conf_chart)
        conf_panel.setLayout(conf_layout)

        infer_panel = QGroupBox("推理耗时折线图")
        infer_layout = QVBoxLayout()
        infer_layout.setContentsMargins(6, 6, 6, 6)
        infer_layout.addWidget(self.infer_chart)
        infer_panel.setLayout(infer_layout)

        stage_panel = QGroupBox("阶段状态灯")
        stage_layout = QVBoxLayout()
        stage_layout.setContentsMargins(6, 6, 6, 6)
        stage_layout.addWidget(self.stage_indicator)
        stage_panel.setLayout(stage_layout)

        visual_grid.addWidget(conf_panel, 0, 0)
        visual_grid.addWidget(infer_panel, 0, 1)
        visual_grid.addWidget(stage_panel, 1, 0, 1, 2)
        visual_box.setLayout(visual_grid)

        cfg_layout = QVBoxLayout()
        cfg_layout.addLayout(top_form)
        cfg_layout.addLayout(middle)
        cfg_layout.addWidget(self._hbox(self.btn_start, self.btn_stop, self.btn_view_log))
        cfg_layout.addWidget(visual_box)
        cfg_box.setLayout(cfg_layout)

        right = QVBoxLayout()
        right.addWidget(cfg_box)
        right.addStretch(1)

        root = QHBoxLayout()
        root.addLayout(left, 3)
        root.addLayout(right, 2)
        self.setLayout(root)


    # 摄像头扫描与状态灯

    def _scan_cameras_on_startup(self):
        self.available_cameras = {}
        self.cam_select.clear()

        available_list = []
        for idx in range(self.camera_scan_range):
            ok = self._test_camera_index(idx)
            self.available_cameras[idx] = ok
            cam_text = f"{'🟢' if ok else '🔴'} 摄像头 {idx}"
            self.cam_select.addItem(cam_text, idx)
            if ok:
                available_list.append(str(idx))

        if self.cam_select.count() == 0:
            self.cam_select.addItem("无可用摄像头", -1)
            self.status.setText("未扫描到可用摄像头")
            self.log_manager.warning("未扫描到可用摄像头")
        else:
            self.status.setText("摄像头扫描完成")
            self.log_manager.info(f"摄像头扫描完成，可用摄像头：{', '.join(available_list) if available_list else '无'}")

    # 测试指定摄像头索引是否可正常打开。
    def _test_camera_index(self, idx: int) -> bool:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            return False

        ok_frame = False
        for _ in range(3):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                if h > 0 and w > 0:
                    ok_frame = True
                    break

        cap.release()
        return ok_frame

    # 清空运行期的图表、阶段和进度显示。
    def _reset_visual_widgets(self):
        self.conf_chart.clear()
        self.conf_chart.set_threshold(self.rule.conf_th)
        self.infer_chart.clear()
        self.hits_progress.set_values(0, self.rule.display_hits_required, self.rule.hits_required)
        self.stage_indicator.set_stage_state("无", 0)


    # 通用小模块 横向布局
    def _hbox(self, *widgets):
        box = QHBoxLayout()
        w = QWidget()
        for it in widgets:
            box.addWidget(it)
        w.setLayout(box)
        return w

    # 选择模型文件并更新界面显示。
    def pick_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", os.path.dirname(DEFAULT_MODEL_PATH), "PyTorch (*.pt)")
        if path:
            self.model_path_label.setText(path)
            self.log_manager.info(f"选择模型文件：{path}")

    # 选择视频文件并更新界面显示。
    def pick_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", os.path.dirname(DEFAULT_VIDEO_PATH), "Video (*.mp4 *.avi *.mkv *.mov)")
        if path:
            self.video_path_label.setText(path)
            self.log_manager.info(f"选择视频文件：{path}")

    # 选择日志保存目录，并同步更新日志管理器。
    def pick_log_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择日志保存目录", self.log_dir)
        if path:
            self.log_dir = path
            self.log_dir_label.setText(path)
            self.log_manager.set_log_dir(path)
            self.log_manager.info(f"日志目录已设置为：{path}")

    # 打开日志查看窗口。
    def show_log_viewer(self):
        self.log_viewer.show()
        self.log_viewer.raise_()
        self.log_viewer.activateWindow()

    # 重置一次运行过程中的统计量。
    def _reset_run_stats(self):
        self.run_start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.run_stats = {
            "total_frames": 0,
            "infer_sum_ms": 0.0,
            "infer_max_ms": 0.0,
            "total_hits": 0,
            "total_triggers": 0
        }

    # 收集界面参数，创建日志文件和工作线程，并启动检测。
    def start(self):
        if self.worker is not None:
            return

        # 读取 UI 参数 -> rule
        self.rule.conf_th = float(self.conf_th.value())
        self.rule.hits_required = int(self.hits_required.value())
        self.rule.display_hits_required = int(self.display_hits_required.value())
        self.rule.hit_window_sec = float(self.hit_window_sec.value())
        self.rule.min_area_ratio = float(self.min_area_ratio.value())
        self.rule.cooldown_sec = float(self.cooldown_sec.value())
        self.rule.imgsz = int(self.imgsz.value())
        self.rule.iou = float(self.iou.value())
        self.rule.display_hits_required = min(self.rule.display_hits_required, self.rule.hits_required)

        selected_device = self.device_select.currentData()
        if selected_device == "0" and (not HAS_CUDA):
            QMessageBox.information(
                self,
                "提示",
                "你选择了 GPU (device=0)，但当前环境未检测到可用 CUDA。\n系统将自动回退为 CPU 继续运行。"
            )
            self.device_select.setCurrentIndex(1)
            selected_device = "cpu"
            self.log_manager.warning("当前环境未检测到可用 CUDA，系统自动回退为 CPU")
        self.rule.device = selected_device

        model_path = self.model_path_label.text().strip()
        if not model_path:
            QMessageBox.warning(self, "提示", "请先选择模型文件")
            self.log_manager.warning("开始运行失败：未选择模型文件")
            return

        source_mode = "video" if self.rb_video.isChecked() else "camera"
        video_path = self.video_path_label.text().strip()
        cam_index = int(self.cam_select.currentData())

        if source_mode == "video" and (not video_path or not os.path.exists(video_path)):
            QMessageBox.warning(self, "提示", "视频路径无效，请重新选择视频文件")
            self.log_manager.warning("开始运行失败：视频路径无效")
            return

        if source_mode == "camera":
            if cam_index < 0:
                QMessageBox.warning(self, "提示", "当前没有可用摄像头，请先重新扫描。")
                self.log_manager.warning("开始运行失败：没有可用摄像头")
                return

            ok = self._test_camera_index(cam_index)
            if not ok:
                QMessageBox.warning(
                    self,
                    "提示",
                    f"摄像头 {cam_index} 当前不可用或取帧异常。\n请重新扫描后选择其他可用摄像头。"
                )
                self._scan_cameras_on_startup()
                self.log_manager.warning(f"开始运行失败：摄像头 {cam_index} 当前不可用")
                return

        self._reset_visual_widgets()
        self._reset_run_stats()
        self.log_manager.start_new_run()
        self.log_manager.info("开始运行")
        self.log_manager.info(f"输入源：{source_mode}")
        self.log_manager.info(f"模型路径：{model_path}")
        if source_mode == "video":
            self.log_manager.info(f"视频路径：{video_path}")
        else:
            self.log_manager.info(f"摄像头索引：{cam_index}")

        self.worker = VideoWorker(model_path, source_mode, video_path, cam_index, self.rule)
        self.worker.frame_signal.connect(self.on_frame)
        self.worker.status_signal.connect(self.on_status)
        self.worker.notify_signal.connect(self.on_notify)
        self.worker.info_signal.connect(self.on_info)
        self.worker.stopped_signal.connect(self.on_stopped)
        self.worker.finished_signal.connect(self.on_worker_finished)
        self.worker.system_log_signal.connect(self.on_system_log)
        self.worker.event_log_signal.connect(self.on_event_log)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        if self.rule.device == "0":
            self.status.setText("启动中... 当前设备：GPU (device=0)")
        else:
            self.status.setText("启动中... 当前设备：CPU")
        self.worker.start()

    # 请求停止当前检测线程。
    def stop(self):
        if self.worker:
            self.worker.stop()
            self.btn_stop.setEnabled(False)
            self.status.setText("停止中...")
            self.log_manager.info("收到停止请求")

    # 接收线程传来的图像帧并更新显示。
    def on_frame(self, qimg: QImage):
        # 自适应缩放
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # 更新底部状态文本。
    def on_status(self, s: str):
        self.status.setText(s)

    # 将系统日志写入日志模块。
    def on_system_log(self, level: str, message: str):
        if level == "INFO":
            self.log_manager.info(message)
        elif level == "WARNING":
            self.log_manager.warning(message)
        elif level == "ERROR":
            self.log_manager.error(message)
        else:
            self.log_manager.log(level, message)

    # 将事件日志写入事件 CSV。
    def on_event_log(self, row: dict):
        self.log_manager.event(row)

    # 接收线程运行信息并刷新统计面板。
    def on_info(self, info: dict):
        self.info_frame.setText(str(info.get("frame_id", 0)))
        self.info_fps.setText(f'{info.get("fps", 0.0):.1f}')
        self.info_raw.setText(str(info.get("raw_count", 0)))
        self.info_hit.setText(str(info.get("valid_count", 0)))

        self.conf_chart.append_value(info.get("best_conf", 0.0))
        self.infer_chart.append_value(info.get("infer_ms", 0.0))
        self.hits_progress.set_values(
            info.get("hits", 0),
            self.rule.display_hits_required,
            self.rule.hits_required
        )
        self.stage_indicator.set_stage_state(
            info.get("stage", "无"),
            info.get("triggered", 0)
        )

        self.run_stats["total_frames"] = int(info.get("frame_id", 0))
        self.run_stats["infer_sum_ms"] += float(info.get("infer_ms", 0.0))
        self.run_stats["infer_max_ms"] = max(self.run_stats["infer_max_ms"], float(info.get("infer_ms", 0.0)))
        self.run_stats["total_hits"] += int(info.get("hit_this_frame", 0))
        self.run_stats["total_triggers"] += int(info.get("triggered", 0))

    # 通过托盘气泡显示报警通知。
    def on_notify(self, title: str, msg: str):
        try:
            self.tray.showMessage(title, msg, QSystemTrayIcon.Warning, 2000)
        except Exception:
            pass

    # 线程停止后恢复按钮状态。
    def on_stopped(self):
        pass

    # 在线程自然结束时写入本次运行汇总信息。
    def on_worker_finished(self):
        model_path = self.model_path_label.text().strip()
        source_mode = "video" if self.rb_video.isChecked() else "camera"
        video_path = self.video_path_label.text().strip()
        cam_index = int(self.cam_select.currentData()) if self.cam_select.count() > 0 else -1

        total_frames = self.run_stats["total_frames"]
        avg_infer_ms = self.run_stats["infer_sum_ms"] / total_frames if total_frames > 0 else 0.0

        self.log_manager.summary({
            "run_start_time": self.run_start_time,
            "run_end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": model_path,
            "source_mode": source_mode,
            "video_path": video_path,
            "cam_index": cam_index,
            "conf_th": self.rule.conf_th,
            "hits_required": self.rule.hits_required,
            "display_hits_required": self.rule.display_hits_required,
            "hit_window_sec": self.rule.hit_window_sec,
            "min_area_ratio": self.rule.min_area_ratio,
            "cooldown_sec": self.rule.cooldown_sec,
            "imgsz": self.rule.imgsz,
            "iou": self.rule.iou,
            "device": self.rule.device,
            "total_frames": total_frames,
            "avg_infer_ms": f"{avg_infer_ms:.4f}",
            "max_infer_ms": f"{self.run_stats['infer_max_ms']:.4f}",
            "total_hits": self.run_stats["total_hits"],
            "total_triggers": self.run_stats["total_triggers"]
        })

        self.log_manager.info("运行汇总已写入 summary.csv")

        if self.worker:
            self.worker.wait()
            self.worker.deleteLater()
            self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # 窗口关闭时，确保线程与日志写入线程被正确结束。
    def closeEvent(self, event):
        try:
            if self.worker:
                self.worker.stop()
                self.worker.wait()
            self.log_manager.info("程序退出")
            self.log_manager.stop()
        except Exception:
            pass
        event.accept()


# 程序入口
def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon())  
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
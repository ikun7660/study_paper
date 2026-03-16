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

# 标准库：路径/进程参数/时间戳等
import os
import sys
import time
from dataclasses import dataclass
from collections import deque

# OpenCV：读取视频/摄像头，绘制框与文字，颜色空间转换
import cv2

# -----------------------------
# 路线A：你可以在这里手动写死路径（方式1）
# -----------------------------
# 默认模型路径：可直接修改为本机 best.pt 的绝对路径
# 界面选择模型时会覆盖该默认值
DEFAULT_MODEL_PATH = r"E:\User\ultralytics-8.3.241\runs\detect\yolov8m_150_no_copy_paste\weights\best.pt"
# 默认视频路径：可直接修改为本机视频文件路径
DEFAULT_VIDEO_PATH = r"E:\User\ultralytics-8.3.241\video\video_1.mp4"

# Windows 提示音（非阻塞）
# winsound 仅在 Windows 环境可用；失败时禁用提示音
try:
    import winsound
    HAS_WINSOUND = True
except Exception:
    HAS_WINSOUND = False

# ultralytics
# Ultralytics YOLO：加载 .pt 权重并进行推理
from ultralytics import YOLO

# PyQt5
# PyQt5：UI 主线程 + QThread 后台推理线程
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QRadioButton, QButtonGroup, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QMessageBox, QSystemTrayIcon
)


# 规则/推理参数集中在 RuleConfig，便于 UI 读写与实验对比
@dataclass
class RuleConfig:
    # 规则阈值
    # 规则置信度阈值：低于该值的检测框不计入命中
    conf_th: float = 0.70          # 置信度阈值（命中阈值）
    # 命中次数阈值：窗口内命中次数达到该值才触发
    hits_required: int = 3         # 命中需要的 hit 数（连续/累计窗口内）
    # 命中统计窗口：仅统计最近 hit_window_sec 秒内的命中
    hit_window_sec: float = 1.0    # 统计窗口（秒）
    # 面积比例阈值：过滤过小的框（框面积/图像面积）
    min_area_ratio: float = 0.00   # 最小框面积比例（框面积/图像面积），过滤小噪点框
    # 冷却时间：触发后 cooldown_sec 内不重复触发
    cooldown_sec: float = 1.0      # 触发后冷却时间（秒），防止连发

    # 推理参数
    # 推理输入尺寸：影响速度与精度
    imgsz: int = 640
    # NMS IoU 阈值：控制框合并强度
    iou: float = 0.5
    # 推理输出下限：设低以保留候选框，再用 conf_th 二次过滤
    raw_conf: float = 0.001        # 模型输出的最低 conf（尽量低，避免你“看不到”潜在命中）
    # 推理设备：GPU 用 '0'，CPU 用 'cpu'
    device: str = "0"              # "0" GPU / "cpu"
    # 单帧最大候选框数量：限制推理输出规模
    max_det: int = 300

    # 提示
    enable_sound: bool = True
    enable_notify: bool = True


# VideoWorker：后台线程执行视频读取与推理，避免阻塞 UI
class VideoWorker(QThread):
    # frame_signal：发送 QImage 到 UI 线程显示
    frame_signal = pyqtSignal(QImage)          # 给界面显示
    # status_signal：发送运行状态字符串到 UI
    status_signal = pyqtSignal(str)            # 状态栏
    # notify_signal：触发通知（标题、内容）
    notify_signal = pyqtSignal(str, str)       # (title, msg)
    # stopped_signal：线程结束后通知 UI 复位按钮状态
    stopped_signal = pyqtSignal()

    # 初始化：保存配置，建立命中统计队列与触发时间戳
    def __init__(self, model_path: str, source_mode: str, video_path: str, cam_index: int, rule: RuleConfig):
        super().__init__()
        self.model_path = model_path
        self.source_mode = source_mode  # "video" or "camera"
        self.video_path = video_path
        self.cam_index = cam_index
        self.rule = rule

        self._stop_flag = False

        # hits 统计：保存最近窗口内的 hit 时间戳
        # hit_times：保存最近命中时间戳，用于滑动窗口计数
        self.hit_times = deque()
        # last_trigger_time：上一次触发时间，用于冷却控制
        self.last_trigger_time = 0.0

    # stop：设置停止标志，由 run 循环检查后退出
    def stop(self):
        self._stop_flag = True

    # 提示音：使用 winsound 异步播放，不阻塞推理循环
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

    # 冷却判定：now 与 last_trigger_time 的时间差是否小于 cooldown_sec
    def _within_cooldown(self, now: float) -> bool:
        return (now - self.last_trigger_time) < self.rule.cooldown_sec

    # 记录命中：追加时间戳并清理窗口外记录
    def _push_hit(self, now: float):
        # 记录一次 hit
        self.hit_times.append(now)
        # 清理窗口外的 hit
        window = self.rule.hit_window_sec
        while self.hit_times and (now - self.hit_times[0]) > window:
            self.hit_times.popleft()

    # 命中计数：返回当前窗口内命中次数
    def _hit_count(self, now: float) -> int:
        window = self.rule.hit_window_sec
        while self.hit_times and (now - self.hit_times[0]) > window:
            self.hit_times.popleft()
        return len(self.hit_times)

    # run：线程入口，包含模型加载、视频读取、推理、规则判定、可视化与信号发送
    def run(self):
        # 1) 加载模型
        # 模型路径检查：防止加载不存在的权重文件
        if not os.path.exists(self.model_path):
            self.status_signal.emit(f"模型不存在：{self.model_path}")
            self.stopped_signal.emit()
            return

        try:
            # 加载 YOLO 权重：会在首次推理前初始化模型
            model = YOLO(self.model_path)
        except Exception as e:
            self.status_signal.emit(f"模型加载失败：{e}")
            self.stopped_signal.emit()
            return

        # 2) 打开视频源
        # 视频源选择：视频文件或摄像头二选一
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

        # 读取源 FPS：用于把 frame_id 转换为时间戳显示
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 部分视频源可能不返回 FPS，低值时使用默认 30FPS
        if fps <= 1e-3:
            fps = 30.0

        frame_id = 0
        t0 = time.time()

        # 主循环：逐帧读取并推理，直到停止或读完
        while not self._stop_flag:
            # 读取一帧：ok=False 表示流结束或读取失败
            ok, frame = cap.read()
            if not ok:
                break

            frame_id += 1
            # 图像尺寸：用于计算框面积比例与生成 UI 图像
            h, w = frame.shape[:2]
            # 整帧面积：用于 area_ratio=box_area/img_area
            img_area = float(w * h)

            # 3) 推理（raw_conf 设低，之后用规则 conf_th 再过滤）
            # 推理计时：用于统计单帧推理耗时 infer_ms
            t_infer0 = time.time()
            try:
                # 推理：raw_conf 设低，保留候选框，后续由规则层过滤
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
            # infer_ms：推理耗时（毫秒）
            infer_ms = (time.time() - t_infer0) * 1000.0

            # 4) 从结果里找“本帧最可信的候选框”
            # best：当前帧通过规则过滤后的最优候选框（按 conf 最大）
            best = None  # (conf, xyxy, cls)
            # raw_count：记录模型原始输出框数量（未过滤）
            raw_count = 0

            # results[0]：当前帧的推理结果对象
            r0 = results[0]
            if r0.boxes is not None and len(r0.boxes) > 0:
                # 遍历每个候选框：读取 conf/cls/xyxy 并进行规则过滤
                for b in r0.boxes:
                    raw_count += 1
                    # conf：该候选框的置信度分数
                    conf = float(b.conf.item()) if b.conf is not None else 0.0
                    # cls：类别索引；可用于多类模型的区分
                    cls = int(b.cls.item()) if b.cls is not None else -1
                    # xyxy：左上角(x1,y1)与右下角(x2,y2)
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    # area_ratio：框面积占整帧比例，用于过滤过小框
                    area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / img_area

                    # 规则过滤：conf + 面积
                    # 置信度过滤：低置信度框不计入命中
                    if conf < self.rule.conf_th:
                        continue
                    # 面积过滤：过小框不计入命中
                    if area_ratio < self.rule.min_area_ratio:
                        continue

                    # 选择最大 conf 的框作为当前帧 best
                    if best is None or conf > best[0]:
                        best = (conf, (x1, y1, x2, y2), cls, area_ratio)

            # now：本帧处理时间戳，用于窗口计数与冷却判断
            now = time.time()

            # 5) 规则判定：本帧是否 hit
            # hit_this_frame：当前帧是否存在通过过滤的候选框
            hit_this_frame = 1 if best is not None else 0
            if hit_this_frame:
                self._push_hit(now)

            # hits：窗口内命中次数
            hits = self._hit_count(now)

            # 6) 触发条件：hits 达标 + 不在 cooldown
            triggered = 0
            # 触发条件：命中次数达标且不处于冷却时间
            if hits >= self.rule.hits_required and (not self._within_cooldown(now)):
                triggered = 1
                # 更新触发时间戳：用于后续冷却控制
                self.last_trigger_time = now

                # 非阻塞提示：托盘气泡 + 音效
                if self.rule.enable_notify:
                    # notify_signal：发送托盘通知内容到 UI
                    self.notify_signal.emit(
                        "刀具预警",
                        f"规则命中：conf≥{self.rule.conf_th:.2f}, hits={hits}/{self.rule.hits_required}, area≥{self.rule.min_area_ratio:.3f}"
                    )
                self._play_sound_non_block()

            # 7) 可视化：只画“命中框”
            # vis：用于绘制叠加信息的可视化帧，不直接修改原始 frame
            vis = frame.copy()

            # 命中时绘制：仅显示通过规则过滤后的 best 框
            if best is not None:
                conf, (x1, y1, x2, y2), cls, area_ratio = best
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # 命中框（仅命中时画）
                # 绘制矩形框：使用 BGR 颜色 (0,0,255) 表示红色
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"HIT conf={conf:.2f} area={area_ratio:.3f}"
                # 绘制文字：显示 conf 与 area_ratio 便于调参
                cv2.putText(vis, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 8) 叠加调试信息（帮助你判断：没检测到 vs 被规则过滤）
            # 你之前看到“一堆框”，那是 raw 全部输出；这里我们只显示命中框，但保留统计信息。
            elapsed = time.time() - t0
            t_sec = frame_id / fps
            # info1：帧号/时间/FPS/推理耗时等统计信息
            info1 = f"frame {frame_id}  t={t_sec:.2f}s  fps={fps:.1f}  infer={infer_ms:.1f}ms"
            # info2：原始框数量/命中/窗口计数/触发/剩余冷却等信息
            info2 = f"raw_boxes={raw_count}  hit={hit_this_frame}  hits={hits}/{self.rule.hits_required}  trig={triggered}  cd={max(0.0, self.rule.cooldown_sec-(now-self.last_trigger_time)):.1f}s"
            # info3：当前规则阈值配置的快照
            info3 = f"CONF_TH={self.rule.conf_th:.2f}  MIN_AREA={self.rule.min_area_ratio:.3f}  WIN={self.rule.hit_window_sec:.1f}s"
            y0 = 25
            # 叠加三行调试信息：先画白字再画黑边提高可读性
            for s in [info1, info2, info3]:
                cv2.putText(vis, s, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(vis, s, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)
                y0 += 25

            # 9) 发给 UI 显示
            # OpenCV 默认 BGR；Qt 显示需要 RGB
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            # QImage：使用底层缓冲区构造，需 copy() 以避免内存复用问题
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
            # 发送一帧到 UI：跨线程传递前复制数据
            self.frame_signal.emit(qimg.copy())
            # 状态栏文本：用于 UI 实时显示运行指标
            self.status_signal.emit(f"运行中：frame={frame_id}, raw={raw_count}, hit={hit_this_frame}, hits={hits}, trig={triggered}")

            # 控制一下线程节奏（让 UI 更丝滑）
            # 线程让步：降低 UI 卡顿概率（毫秒级）
            self.msleep(1)

        # 释放视频源：结束后关闭 VideoCapture 句柄
        cap.release()
        self.status_signal.emit("已停止")
        self.stopped_signal.emit()


# MainWindow：UI 配置面板 + 视频显示区域 + 启停控制
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("刀具检测预警系统")
        self.resize(1100, 700)

        # worker：后台 VideoWorker 实例；为 None 表示未运行
        self.worker = None
        # rule：保存当前规则配置；start() 时从 UI 读取并更新
        self.rule = RuleConfig()

        # --- 托盘通知（Windows 风格，不阻塞，自动消失）
        # 系统托盘：用于显示通知气泡，不阻塞主线程
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(self.style().standardIcon(self.style().SP_MessageBoxWarning))
        self.tray.setVisible(True)

        # --- 画面显示
        # 视频显示控件：显示 QImage 转换后的画面
        self.video_label = QLabel("点击开始后显示画面")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 500)
        self.video_label.setStyleSheet("background:#111; color:#ddd; border:1px solid #333;")

        # --- 状态
        # 状态控件：显示运行/停止/错误信息
        self.status = QLabel("就绪")
        self.status.setStyleSheet("color:#333;")

        # --- 模型路径（方式2：界面选择）
        self.model_path_label = QLabel(DEFAULT_MODEL_PATH)
        # 选择模型按钮：通过文件对话框选择 .pt 权重文件
        self.btn_pick_model = QPushButton("选择模型best.pt")
        self.btn_pick_model.clicked.connect(self.pick_model)

        # --- 输入源选择
        # 输入源单选：视频文件模式
        self.rb_video = QRadioButton("视频文件")
        # 输入源单选：摄像头模式
        self.rb_cam = QRadioButton("摄像头")
        self.rb_video.setChecked(True)
        # 按钮组：把两个单选按钮归为同组互斥
        self.source_group = QButtonGroup()
        self.source_group.addButton(self.rb_video)
        self.source_group.addButton(self.rb_cam)

        self.video_path_label = QLabel(DEFAULT_VIDEO_PATH)
        # 选择视频按钮：选择待检测的视频文件
        self.btn_pick_video = QPushButton("选择视频")
        self.btn_pick_video.clicked.connect(self.pick_video)

        # 摄像头索引：对应 cv2.VideoCapture(index)
        self.cam_index = QSpinBox()
        self.cam_index.setRange(0, 10)
        self.cam_index.setValue(0)

        # --- 规则参数
        # 规则参数输入：conf_th
        self.conf_th = QDoubleSpinBox()
        self.conf_th.setRange(0.0, 1.0)
        self.conf_th.setSingleStep(0.01)
        self.conf_th.setValue(self.rule.conf_th)

        # 规则参数输入：hits_required
        self.hits_required = QSpinBox()
        self.hits_required.setRange(1, 30)
        self.hits_required.setValue(self.rule.hits_required)

        # 规则参数输入：hit_window_sec
        self.hit_window_sec = QDoubleSpinBox()
        self.hit_window_sec.setRange(0.1, 10.0)
        self.hit_window_sec.setSingleStep(0.1)
        self.hit_window_sec.setValue(self.rule.hit_window_sec)

        # 规则参数输入：min_area_ratio
        self.min_area_ratio = QDoubleSpinBox()
        self.min_area_ratio.setRange(0.0, 1.0)
        self.min_area_ratio.setSingleStep(0.001)
        self.min_area_ratio.setDecimals(3)
        self.min_area_ratio.setValue(self.rule.min_area_ratio)

        # 规则参数输入：cooldown_sec
        self.cooldown_sec = QDoubleSpinBox()
        self.cooldown_sec.setRange(0.0, 10.0)
        self.cooldown_sec.setSingleStep(0.1)
        self.cooldown_sec.setValue(self.rule.cooldown_sec)

        # --- 推理参数
        # 推理参数输入：imgsz
        self.imgsz = QSpinBox()
        self.imgsz.setRange(320, 1280)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(self.rule.imgsz)

        # 推理参数输入：iou
        self.iou = QDoubleSpinBox()
        self.iou.setRange(0.1, 0.95)
        self.iou.setSingleStep(0.05)
        self.iou.setValue(self.rule.iou)

        self.device_label = QLabel("[device: 0(GPU) 或 cpu]:")
        self.device_value = QLabel(self.rule.device)

        # --- 控制按钮
        # 开始按钮：创建并启动 VideoWorker 线程
        self.btn_start = QPushButton("开始")
        # 停止按钮：设置停止标志，线程自行退出
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)

        # --- 布局
        left = QVBoxLayout()
        left.addWidget(self.video_label)
        left.addWidget(self.status)

        cfg_box = QGroupBox("配置")
        # 表单布局：左侧标签 + 右侧控件
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

        # 根布局：左侧画面区域，右侧配置面板
        root = QHBoxLayout()
        root.addLayout(left, 3)
        root.addLayout(right, 1)
        self.setLayout(root)

    # 辅助布局：把多个控件放到一行返回 QWidget
    def _hbox(self, *widgets):
        box = QHBoxLayout()
        w = QWidget()
        for it in widgets:
            box.addWidget(it)
        w.setLayout(box)
        return w

    # 选择模型：更新 model_path_label 文本
    def pick_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", os.path.dirname(DEFAULT_MODEL_PATH), "PyTorch (*.pt)")
        if path:
            self.model_path_label.setText(path)

    # 选择视频：更新 video_path_label 文本
    def pick_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", os.path.dirname(DEFAULT_VIDEO_PATH), "Video (*.mp4 *.avi *.mkv *.mov)")
        if path:
            self.video_path_label.setText(path)

    # start：读取 UI 参数 -> RuleConfig，并启动后台线程
    def start(self):
        # 重复启动保护：运行中不重复创建线程
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

        # 读取模型路径：来自默认值或文件对话框选择
        model_path = self.model_path_label.text().strip()
        if not model_path:
            QMessageBox.warning(self, "提示", "请先选择模型")
            return

        # 确定输入源模式：根据单选按钮状态
        source_mode = "video" if self.rb_video.isChecked() else "camera"
        video_path = self.video_path_label.text().strip()
        cam_index = int(self.cam_index.value())

        # 视频文件校验：路径为空或文件不存在则提示并返回
        if source_mode == "video" and (not video_path or not os.path.exists(video_path)):
            QMessageBox.warning(self, "提示", "视频路径无效，请选择视频文件")
            return

        # 创建后台线程：传入模型/输入源/规则配置
        self.worker = VideoWorker(model_path, source_mode, video_path, cam_index, self.rule)
        # 连接信号：后台帧 -> UI 刷新画面
        self.worker.frame_signal.connect(self.on_frame)
        # 连接信号：后台状态 -> UI 状态文本
        self.worker.status_signal.connect(self.on_status)
        # 连接信号：后台触发 -> UI 托盘通知
        self.worker.notify_signal.connect(self.on_notify)
        # 连接信号：线程退出 -> UI 复位按钮
        self.worker.stopped_signal.connect(self.on_stopped)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status.setText("启动中...")
        self.worker.start()

    # stop：调用 worker.stop() 设置停止标志
    def stop(self):
        if self.worker:
            self.worker.stop()
            self.btn_stop.setEnabled(False)

    # on_frame：把 QImage 转换为 QPixmap 并按控件尺寸缩放显示
    def on_frame(self, qimg: QImage):
        # 自适应缩放
        # QPixmap：Qt 的显示图像对象
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_status(self, s: str):
        self.status.setText(s)

    # on_notify：通过系统托盘显示通知气泡
    def on_notify(self, title: str, msg: str):
        # Windows 托盘通知（不阻塞、自动消失）
        try:
            self.tray.showMessage(title, msg, QSystemTrayIcon.Warning, 2000)
        except Exception:
            pass

    # on_stopped：线程结束后的 UI 状态复位
    def on_stopped(self):
        self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)


# 程序入口：创建 QApplication 与主窗口并进入事件循环
def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon())  # 你也可以换成自己的 ico
    w = MainWindow()
    w.show()
    # 进入 Qt 事件循环；退出码返回给系统
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

import threading
import time

# Python 标准库 GUI，用于弹窗提示
import tkinter as tk

# ====================== 声音 & 弹窗 ======================
# Windows 自带蜂鸣音（不需要额外 pip 安装）
import winsound
from tkinter import messagebox

import cv2

from ultralytics import YOLO

# ====================== 参数配置区（核心可调） ======================

MODEL_PATH = r"runs\detect\train6\weights\best.pt"  # 你训练好的 YOLO 权重路径

TARGET_CLASS_ID = None
# 只针对某一个类别触发（例如 0 表示“人”）
# None 表示：检测到任何类别都算

CONF_TH = 0.70
# 置信度阈值：模型认为“是目标”的可信程度
# 一般 0.6~0.8 之间调

MIN_HITS = 5
# 连续多少帧检测到目标才触发
# 用来防止单帧误报

COOLDOWN_SEC = 3
# 冷却时间（秒）：触发一次后，多少秒内不再重复触发
# 防止疯狂弹窗 / 响铃

MIN_AREA_RATIO = 0.01
# 目标框面积 / 画面面积
# 过滤很远、很小的目标噪声（可选，但很实用）


# ====================== 模型加载 ======================

# 加载 YOLOv8 模型
model = YOLO(MODEL_PATH)


# ====================== 状态变量 ======================

hit_count = 0
# 当前“连续命中”的帧数

last_trigger_time = 0
# 上一次触发提醒的时间戳（用于冷却时间判断）


# ====================== 弹窗 + 声音（子线程执行） ======================


def popup_and_beep(text: str):
    """弹窗 + 声音提示函数 放在子线程中执行，避免阻塞摄像头主循环.
    """
    try:
        # 蜂鸣音（频率Hz, 时长ms）
        winsound.Beep(1200, 200)
        winsound.Beep(1200, 200)
    except:
        pass

    # tkinter 弹窗
    root = tk.Tk()
    root.withdraw()  # 不显示主窗口
    messagebox.showinfo("检测提示", text)
    root.destroy()


# ====================== 打开摄像头 ======================

cap = cv2.VideoCapture(0)  # 0 表示默认摄像头
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")


# ====================== 主循环（逐帧检测） ======================

while True:
    # 从摄像头读取一帧画面
    ret, frame = cap.read()
    if not ret:
        break

    # 当前帧的尺寸
    H, W = frame.shape[:2]
    frame_area = H * W

    # 使用 YOLO 进行检测 + 跟踪
    # persist=True：启用目标ID跟踪（跨帧）
    results = model.track(frame, persist=True, verbose=False)
    r = results[0]

    detected = False  # 本帧是否满足“有效检测”条件

    # ====================== 遍历检测框 ======================
    if r.boxes is not None and len(r.boxes) > 0:
        # conf: 置信度
        # cls: 类别ID
        # xyxy: 左上右下坐标
        for conf, cls, xyxy in zip(r.boxes.conf, r.boxes.cls, r.boxes.xyxy):
            conf = float(conf)
            cls = int(cls)

            # 如果只关注某个类别，其它类别跳过
            if TARGET_CLASS_ID is not None and cls != TARGET_CLASS_ID:
                continue

            # 计算检测框面积
            x1, y1, x2, y2 = map(float, xyxy)
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_ratio = area / frame_area

            # 核心判定条件：
            # 1. 置信度足够高
            # 2. 目标面积足够大
            if conf >= CONF_TH and area_ratio >= MIN_AREA_RATIO:
                detected = True
                break

    # ====================== 连续命中计数逻辑 ======================

    if detected:
        hit_count += 1  # 本帧满足条件 → 连续命中 +1
    else:
        hit_count = 0  # 中断 → 清零

    # ====================== 结果可视化 ======================

    # 在画面上画检测框
    annotated = r.plot()

    # 左上角显示当前状态信息（调试用）
    cv2.putText(
        annotated,
        f"hit={hit_count}/{MIN_HITS} conf_th={CONF_TH}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if detected else (0, 0, 255),
        2,
    )

    cv2.imshow("cam", annotated)

    # ====================== 触发条件判断 ======================

    now = time.time()

    # 触发条件：
    # 1. 连续命中帧数达到阈值
    # 2. 距离上一次触发超过冷却时间
    if hit_count >= MIN_HITS and (now - last_trigger_time) >= COOLDOWN_SEC:
        last_trigger_time = now
        hit_count = 0  # 触发后清零，防止重复触发

        # 使用子线程弹窗 + 声音
        threading.Thread(
            target=popup_and_beep, args=(f"检测到目标（conf ≥ {CONF_TH}，连续 {MIN_HITS} 帧）",), daemon=True
        ).start()

    # 按 q 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ====================== 资源释放 ======================

cap.release()
cv2.destroyAllWindows()

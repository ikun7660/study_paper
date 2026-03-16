import threading
import time

# ---------- 弹窗（标准库） ----------
import tkinter as tk
from tkinter import messagebox

import cv2

from ultralytics import YOLO

# ---------- 声音（Windows自带，无需安装） ----------
# 若不是 Windows：可以把 winsound 部分注释掉，或改用 playsound/pydub 等库
try:
    import winsound

    HAS_WINSOUND = True
except Exception:
    HAS_WINSOUND = False


# =========================
# 1) 参数区：你主要调这些
# =========================
MODEL_PATH = "shu.pt"  # 你的训练权重文件
CAM_ID = 0  # 摄像头编号：0/1/2...

# 是否只对某个类别触发：例如 0 表示“人”
# None 表示检测到任何类别都触发（不推荐，误触发概率更高）
TARGET_CLASS_ID = None

CONF_TH = 0.70  # 置信度阈值（0~1）
MIN_HITS = 5  # 连续命中帧数（抗抖动）
COOLDOWN_SEC = 3  # 触发冷却时间（秒）
MIN_AREA_RATIO = 0.01  # 目标框面积/画面面积（过滤远小目标）

SHOW_FPS = True  # 是否显示 FPS
USE_TRACK = True  # True: track(跟踪); False: predict(检测)


# =========================
# 2) 工具函数：弹窗+提示音
# =========================
def popup_and_beep(text: str):
    """子线程执行：弹窗 + 蜂鸣音 注意：弹窗是阻塞的，所以必须放子线程，否则视频窗口会卡住。.
    """
    # 蜂鸣音（Windows）
    if HAS_WINSOUND:
        try:
            winsound.Beep(1200, 200)
            winsound.Beep(1200, 200)
        except Exception:
            pass

    # 弹窗（tkinter）
    root = tk.Tk()
    root.withdraw()  # 不显示主窗口
    messagebox.showinfo("检测提示", text)
    root.destroy()


# =========================
# 3) 主程序
# =========================
def main():
    # 加载模型
    model = YOLO(MODEL_PATH)

    # 打开摄像头
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {CAM_ID}")

    # 状态变量
    hit_count = 0
    last_trigger_time = 0.0

    # FPS 统计
    last_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W = frame.shape[:2]
        frame_area = H * W

        # -------------------------
        # 4) YOLO 推理：track 或 predict
        # -------------------------
        if USE_TRACK:
            results = model.track(frame, persist=True, verbose=False)
        else:
            results = model.predict(frame, verbose=False)

        r = results[0]

        # -------------------------
        # 5) 判断本帧是否“有效检测”
        #    有效 = conf>=阈值 且 面积>=阈值 且(可选)类别匹配
        # -------------------------
        detected = False

        if r.boxes is not None and len(r.boxes) > 0:
            for conf, cls, xyxy in zip(r.boxes.conf, r.boxes.cls, r.boxes.xyxy):
                conf = float(conf)
                cls = int(cls)

                # 类别过滤（只关心一个类时强烈建议开）
                if TARGET_CLASS_ID is not None and cls != TARGET_CLASS_ID:
                    continue

                x1, y1, x2, y2 = map(float, xyxy)
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                area_ratio = area / frame_area

                if conf >= CONF_TH and area_ratio >= MIN_AREA_RATIO:
                    detected = True
                    break

        # 连续命中逻辑（抗抖动）
        if detected:
            hit_count += 1
        else:
            hit_count = 0

        # -------------------------
        # 6) 直接显示 YOLO 的“画好框的新图”
        # -------------------------
        annotated = r.plot()  # 关键：这就是你问的“直接显示YOLO返回的新图”

        # 可选：叠加状态信息（不会影响YOLO框，只是额外文本）
        if SHOW_FPS:
            now = time.time()
            dt = now - last_time
            if dt > 0:
                fps = 1.0 / dt
            last_time = now

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}  hit={hit_count}/{MIN_HITS}  conf_th={CONF_TH}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if detected else (0, 0, 255),
                2,
            )

        cv2.imshow("YOLO Camera", annotated)

        # -------------------------
        # 7) 触发：连续命中 + 冷却时间
        # -------------------------
        now = time.time()
        if hit_count >= MIN_HITS and (now - last_trigger_time) >= COOLDOWN_SEC:
            last_trigger_time = now
            hit_count = 0  # 触发一次后清零，避免连环触发

            msg = f"检测到目标（conf ≥ {CONF_TH}，连续 {MIN_HITS} 帧）"
            threading.Thread(target=popup_and_beep, args=(msg,), daemon=True).start()

        # q 退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import os
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# =========================
# 你只需要改这里
# =========================
MODEL_PATH = r"runs\detect\yolov8m_150_no_copy_paste\weights\best.pt"
VIDEO_PATH = r"E:\User\ultralytics-8.3.241\video\vedio_2.mp4"
OUT_ROOT = r"system_eval"

TARGET_CLASS_ID = None


# =========================
# 规则列表
# =========================
RULE_SETS = [
    ("R1_conf0.70_hits1_area0.00_cd0", 0.70, 1, 0.0, 0.00),
    ("R2_conf0.70_hits3_area0.00_cd1", 0.70, 3, 1.0, 0.00),
    ("R3_conf0.70_hits5_area0.00_cd3", 0.70, 5, 3.0, 0.00),
    ("R4_conf0.80_hits5_area0.00_cd3", 0.80, 5, 3.0, 0.00),
    ("R5_conf0.70_hits3_area0.01_cd1", 0.70, 3, 1.0, 0.01),
    ("R6_conf0.70_hits5_area0.01_cd3", 0.70, 5, 3.0, 0.01),
    ("R7_conf0.85_hits5_area0.01_cd3", 0.85, 5, 3.0, 0.01),
]


# =========================
# 数据结构
# =========================
@dataclass
class RuleConfig:
    name: str
    conf_th: float
    min_hits: int
    cooldown_sec: float
    min_area_ratio: float


@dataclass
class FrameRecord:
    rule_name: str
    frame_idx: int
    t_sec: float
    detected: int
    hit_count: int
    triggered: int
    best_conf: float
    best_area_ratio: float
    n_boxes: int
    max_conf_any: float
    mean_conf: float


# =========================
# 工具函数
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default


def read_video_info(cap: cv2.VideoCapture):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return w, h, fps, frame_count


def extract_boxes(r, target_cls):
    out = []
    if r.boxes is None or len(r.boxes) == 0:
        return out

    for conf, cls, box in zip(r.boxes.conf, r.boxes.cls, r.boxes.xyxy):
        c = safe_float(conf)
        k = int(cls)
        if target_cls is not None and k != target_cls:
            continue
        x1, y1, x2, y2 = map(float, box)
        out.append((c, k, (x1, y1, x2, y2)))
    return out


def calc_area_ratio(box, frame_area):
    x1, y1, x2, y2 = box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area / frame_area if frame_area > 0 else 0.0


# =========================
# 核心评估函数
# =========================
def run_one_rule(model, video_path, rule, out_dir):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    W, H, fps, frame_count = read_video_info(cap)
    frame_area = float(W * H)
    duration_sec = frame_count / fps if fps > 0 else 0.0

    hit_count = 0
    last_trigger_t = -1e9
    frames = []

    infer_time_sum = 0.0
    n_infer = 0

    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t_sec = frame_idx / fps if fps > 0 else float(frame_idx)

        t0 = time.time()
        res = model.predict(
            frame,
            imgsz=640,
            conf=0.001,
            iou=0.7,
            verbose=False
        )
        infer_time_sum += (time.time() - t0)
        n_infer += 1

        boxes = extract_boxes(res[0], TARGET_CLASS_ID)

        n_boxes = len(boxes)
        conf_list = [b[0] for b in boxes] if boxes else []
        max_conf_any = max(conf_list) if conf_list else 0.0
        mean_conf = float(np.mean(conf_list)) if conf_list else 0.0

        best_conf = 0.0
        best_area_ratio = 0.0
        detected = 0

        if boxes:
            boxes_sorted = sorted(boxes, key=lambda x: x[0], reverse=True)
            for conf, cls, box in boxes_sorted:
                area_ratio = calc_area_ratio(box, frame_area)
                if conf >= rule.conf_th and area_ratio >= rule.min_area_ratio:
                    detected = 1
                    best_conf = conf
                    best_area_ratio = area_ratio
                    break

        if detected:
            hit_count += 1
        else:
            hit_count = 0

        triggered = 0
        if hit_count >= rule.min_hits and (t_sec - last_trigger_t) >= rule.cooldown_sec:
            triggered = 1
            last_trigger_t = t_sec
            hit_count = 0

        frames.append(FrameRecord(
            rule.name, frame_idx, t_sec, detected, hit_count,
            triggered, best_conf, best_area_ratio,
            n_boxes, max_conf_any, mean_conf
        ))

    cap.release()

    df_frames = pd.DataFrame([asdict(x) for x in frames])
    ensure_dir(out_dir)
    df_frames.to_csv(os.path.join(out_dir, "per_frame.csv"), index=False, encoding="utf-8-sig")

    n_frames = len(df_frames)
    n_detected = int(df_frames["detected"].sum())
    n_triggers = int(df_frames["triggered"].sum())

    triggers_per_min = (n_triggers / (duration_sec / 60.0)) if duration_sec > 0 else 0.0
    detected_ratio = (n_detected / n_frames) if n_frames > 0 else 0.0

    avg_infer_ms = (infer_time_sum / n_infer * 1000.0) if n_infer > 0 else 0.0
    eff_fps = (n_infer / infer_time_sum) if infer_time_sum > 0 else 0.0

    summary = {
        "rule_name": rule.name,
        "conf_th": rule.conf_th,
        "min_hits": rule.min_hits,
        "cooldown_sec": rule.cooldown_sec,
        "min_area_ratio": rule.min_area_ratio,
        "trigger_count": n_triggers,
        "triggers_per_min": triggers_per_min,
        "detected_ratio": detected_ratio,
        "avg_infer_ms": avg_infer_ms,
        "effective_fps": eff_fps
    }

    return summary


# =========================
# 主程序
# =========================
def main():
    ensure_dir(OUT_ROOT)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(OUT_ROOT, f"video_eval_{timestamp}")
    ensure_dir(run_root)

    model = YOLO(MODEL_PATH)

    all_summaries = []
    for name, conf_th, min_hits, cooldown, min_area_ratio in RULE_SETS:
        rule = RuleConfig(name, conf_th, min_hits, cooldown, min_area_ratio)
        out_dir = os.path.join(run_root, rule.name)
        summary = run_one_rule(model, VIDEO_PATH, rule, out_dir)
        all_summaries.append(summary)

    df_all = pd.DataFrame(all_summaries)
    df_all.to_csv(os.path.join(run_root, "ALL_summary.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

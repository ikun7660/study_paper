# -*- coding: utf-8 -*-
"""
system_eval_compare.py

作用：
1. 用同一批视频分别评估：
   - baseline：原始YOLO直接报警（无候选/命中累计/冷却）
   - final：当前完整系统逻辑（与 knife_alarm_gui.py 中 VideoWorker.run 的规则一致）
2. 输出 summary.csv、per_frame.csv
3. 自动保存一两张鲜明对比截图（baseline vs final）

说明：
- final 模式的判定逻辑尽量按你当前系统保持一致
- 去掉了 GUI、托盘通知、日志窗口，仅保留实验所需逻辑
"""

import os
import csv
import time
from dataclasses import dataclass
from collections import deque
from typing import List, Tuple, Dict, Any

import cv2
from ultralytics import YOLO

try:
    import torch
    HAS_CUDA = bool(torch.cuda.is_available())
except Exception:
    HAS_CUDA = False


# =========================
# 1. 这里改成你的路径
# =========================
MODEL_PATH = r"E:\User\ultralytics-8.3.241\runs\detect\yolov8m_150_no_copy_paste\weights\best.pt"

VIDEO_LIST = [
    # (视频路径, 视频类别)
    (r"E:\User\ultralytics-8.3.241\video\有刀.mp4", "positive"),
    (r"E:\User\ultralytics-8.3.241\video\无刀.mp4", "negative"),
    (r"E:\User\ultralytics-8.3.241\video\模糊干扰.mp4", "hard"),
]

OUT_DIR = r"E:\User\system_eval_compare"


# =========================
# 2. 规则配置（按你系统来）
# =========================
@dataclass
class RuleConfig:
    conf_th: float = 0.50
    hits_required: int = 15
    hit_window_sec: float = 1.0
    min_area_ratio: float = 0.00
    cooldown_sec: float = 1.0

    imgsz: int = 640
    iou: float = 0.5
    raw_conf: float = 0.001
    device: str = "0" if HAS_CUDA else "cpu"
    max_det: int = 300

    display_hits_required: int = 10

    enable_sound: bool = False
    enable_notify: bool = False


RULE = RuleConfig()


# =========================
# 3. 工具函数
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_video_name(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def draw_boxes_for_stage(frame, valid_boxes, stage: str):
    """
    按你当前系统的显示方式画框：
    - 候选：黄色细框
    - 确认：红色细框
    - 报警：红色粗框
    """
    vis = frame.copy()
    for item in valid_boxes:
        conf, (x1, y1, x2, y2), cls, area_ratio = item
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        if stage == "报警":
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        elif stage == "确认":
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        elif stage == "候选":
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
    return vis


def draw_baseline_boxes(frame, valid_boxes):
    """
    baseline 显示：只要过 conf_th 的框都画出来，统一红框
    """
    vis = frame.copy()
    for item in valid_boxes:
        conf, (x1, y1, x2, y2), cls, area_ratio = item
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return vis


def put_label(frame, text: str):
    out = frame.copy()
    cv2.putText(out, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return out


# =========================
# 4. final 模式：保持你系统逻辑
# =========================
class FinalEvaluator:
    def __init__(self, rule: RuleConfig):
        self.rule = rule
        self.hit_times = deque()
        self.last_trigger_time = 0.0
        self.prev_stage = "无"

    def _within_cooldown(self, now: float) -> bool:
        return (now - self.last_trigger_time) < self.rule.cooldown_sec

    def _push_hit(self, now: float):
        self.hit_times.append(now)
        window = self.rule.hit_window_sec
        while self.hit_times and (now - self.hit_times[0]) > window:
            self.hit_times.popleft()

    def _hit_count(self, now: float) -> int:
        window = self.rule.hit_window_sec
        while self.hit_times and (now - self.hit_times[0]) > window:
            self.hit_times.popleft()
        return len(self.hit_times)

    def step(self, frame, results0, frame_id: int) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        img_area = float(w * h)

        valid_boxes = []
        best = None
        raw_count = 0

        if results0.boxes is not None and len(results0.boxes) > 0:
            for b in results0.boxes:
                raw_count += 1
                conf = float(b.conf.item()) if b.conf is not None else 0.0
                cls = int(b.cls.item()) if b.cls is not None else -1
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / img_area

                # 与你当前系统一致：conf + 面积过滤
                if conf < self.rule.conf_th:
                    continue
                if area_ratio < self.rule.min_area_ratio:
                    continue

                item = (conf, (x1, y1, x2, y2), cls, area_ratio)
                valid_boxes.append(item)

                if best is None or conf > best[0]:
                    best = item

        now = time.time()

        # 与你当前系统一致：单帧只要存在至少一个有效框，就记 1 次 hit
        hit_this_frame = 1 if len(valid_boxes) > 0 else 0
        if hit_this_frame:
            self._push_hit(now)

        hits = self._hit_count(now)

        # 与你当前系统一致：hits 达标 + 不在 cooldown -> triggered
        triggered = 0
        if hits >= self.rule.hits_required and (not self._within_cooldown(now)):
            triggered = 1
            self.last_trigger_time = now

        display_candidate = len(valid_boxes) > 0
        display_confirmed = display_candidate and (hits >= self.rule.display_hits_required)
        display_alert = display_confirmed and (triggered == 1 or self._within_cooldown(now))

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

        vis = draw_boxes_for_stage(frame, valid_boxes, stage)
        vis = put_label(
            vis,
            f"FINAL | stage={stage} raw={raw_count} valid={len(valid_boxes)} hits={hits} trig={triggered}"
        )

        return {
            "frame_id": frame_id,
            "raw_count": raw_count,
            "valid_count": len(valid_boxes),
            "hit_this_frame": hit_this_frame,
            "hits": hits,
            "triggered": triggered,
            "stage": stage,
            "best_conf": best_conf,
            "best_area_ratio": best_area_ratio,
            "vis": vis,
        }


# =========================
# 5. baseline 模式：原始YOLO直接报警
# =========================
class BaselineEvaluator:
    def __init__(self, rule: RuleConfig):
        self.rule = rule

    def step(self, frame, results0, frame_id: int) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        img_area = float(w * h)

        valid_boxes = []
        raw_count = 0
        best_conf = 0.0
        best_area_ratio = 0.0

        if results0.boxes is not None and len(results0.boxes) > 0:
            for b in results0.boxes:
                raw_count += 1
                conf = float(b.conf.item()) if b.conf is not None else 0.0
                cls = int(b.cls.item()) if b.cls is not None else -1
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / img_area

                # baseline：只做 conf_th 过滤，不做 hits/cooldown/阶段机制
                if conf < self.rule.conf_th:
                    continue

                item = (conf, (x1, y1, x2, y2), cls, area_ratio)
                valid_boxes.append(item)

                if conf > best_conf:
                    best_conf = conf
                    best_area_ratio = area_ratio

        # baseline：这一帧只要有有效框，就直接算报警
        triggered = 1 if len(valid_boxes) > 0 else 0
        stage = "报警" if triggered == 1 else "无"

        vis = draw_baseline_boxes(frame, valid_boxes)
        vis = put_label(
            vis,
            f"BASELINE | raw={raw_count} valid={len(valid_boxes)} trig={triggered}"
        )

        return {
            "frame_id": frame_id,
            "raw_count": raw_count,
            "valid_count": len(valid_boxes),
            "hit_this_frame": 1 if len(valid_boxes) > 0 else 0,
            "hits": 0,
            "triggered": triggered,
            "stage": stage,
            "best_conf": best_conf,
            "best_area_ratio": best_area_ratio,
            "vis": vis,
        }


# =========================
# 6. 评估主函数
# =========================
def run_one_mode(model, video_path: str, video_type: str, mode_name: str, rule: RuleConfig, out_dir: str):
    ensure_dir(out_dir)
    screenshot_dir = os.path.join(out_dir, "screenshots")
    ensure_dir(screenshot_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"视频打不开：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 30.0

    evaluator = FinalEvaluator(rule) if mode_name == "final" else BaselineEvaluator(rule)

    per_frame_rows = []
    frame_id = 0
    infer_sum_ms = 0.0
    infer_max_ms = 0.0
    total_triggers = 0
    total_hits = 0

    # 自动抓图：保存前2张最有代表性的图
    # 规则：
    # - baseline：优先抓到有报警的帧
    # - final：优先抓到“确认/报警”的帧
    saved_shots = 0
    max_shots = 2

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_id += 1

        t_infer0 = time.time()
        results = model.predict(
            source=frame,
            imgsz=rule.imgsz,
            conf=rule.raw_conf,   # 和你当前系统一致：先低阈值推理，再规则过滤
            iou=rule.iou,
            device=rule.device,
            verbose=False,
            max_det=rule.max_det
        )
        infer_ms = (time.time() - t_infer0) * 1000.0
        infer_sum_ms += infer_ms
        infer_max_ms = max(infer_max_ms, infer_ms)

        step_res = evaluator.step(frame, results[0], frame_id)
        total_triggers += int(step_res["triggered"])
        total_hits += int(step_res["hit_this_frame"])

        row = {
            "mode": mode_name,
            "video_name": safe_video_name(video_path),
            "video_type": video_type,
            "frame_id": frame_id,
            "fps": fps,
            "infer_ms": infer_ms,
            "raw_count": step_res["raw_count"],
            "valid_count": step_res["valid_count"],
            "hit_this_frame": step_res["hit_this_frame"],
            "hits": step_res["hits"],
            "triggered": step_res["triggered"],
            "stage": step_res["stage"],
            "best_conf": step_res["best_conf"],
            "best_area_ratio": step_res["best_area_ratio"],
        }
        per_frame_rows.append(row)

        # 自动截图
        should_save = False
        if saved_shots < max_shots:
            if mode_name == "baseline" and step_res["triggered"] == 1:
                should_save = True
            elif mode_name == "final" and step_res["stage"] in ("确认", "报警"):
                should_save = True

        if should_save:
            shot_name = f"{mode_name}_{safe_video_name(video_path)}_frame{frame_id}.jpg"
            shot_path = os.path.join(screenshot_dir, shot_name)
            cv2.imwrite(shot_path, step_res["vis"])
            saved_shots += 1

    cap.release()

    frames = frame_id
    avg_infer_ms = infer_sum_ms / frames if frames > 0 else 0.0
    effective_fps = 1000.0 / avg_infer_ms if avg_infer_ms > 1e-6 else 0.0
    duration_sec = frames / fps if fps > 1e-6 else 0.0
    triggers_per_min = (total_triggers / duration_sec * 60.0) if duration_sec > 1e-6 else 0.0

    candidate_frames = sum(1 for r in per_frame_rows if r["stage"] == "候选")
    confirmed_frames = sum(1 for r in per_frame_rows if r["stage"] == "确认")
    alert_frames = sum(1 for r in per_frame_rows if r["stage"] == "报警")

    summary = {
        "mode": mode_name,
        "video_name": safe_video_name(video_path),
        "video_type": video_type,
        "video_path": video_path,
        "frames": frames,
        "duration_sec": round(duration_sec, 3),
        "fps_meta": round(fps, 3),
        "avg_infer_ms": round(avg_infer_ms, 4),
        "max_infer_ms": round(infer_max_ms, 4),
        "effective_fps": round(effective_fps, 3),
        "total_hits": total_hits,
        "total_triggers": total_triggers,
        "triggers_per_min": round(triggers_per_min, 3),
        "candidate_frames": candidate_frames,
        "confirmed_frames": confirmed_frames,
        "alert_frames": alert_frames,
        "avg_raw_count": round(sum(r["raw_count"] for r in per_frame_rows) / frames, 4) if frames > 0 else 0.0,
        "avg_valid_count": round(sum(r["valid_count"] for r in per_frame_rows) / frames, 4) if frames > 0 else 0.0,
        "max_best_conf": round(max([r["best_conf"] for r in per_frame_rows] + [0.0]), 4),
        "rule_conf_th": rule.conf_th,
        "rule_hits_required": rule.hits_required,
        "rule_display_hits_required": rule.display_hits_required,
        "rule_hit_window_sec": rule.hit_window_sec,
        "rule_min_area_ratio": rule.min_area_ratio,
        "rule_cooldown_sec": rule.cooldown_sec,
        "rule_raw_conf": rule.raw_conf,
        "rule_iou": rule.iou,
        "rule_imgsz": rule.imgsz,
        "rule_device": rule.device,
    }

    # 写 per_frame.csv
    per_frame_csv = os.path.join(out_dir, f"{mode_name}_{safe_video_name(video_path)}_per_frame.csv")
    with open(per_frame_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_frame_rows[0].keys()) if per_frame_rows else [
            "mode", "video_name", "video_type", "frame_id", "fps", "infer_ms",
            "raw_count", "valid_count", "hit_this_frame", "hits", "triggered",
            "stage", "best_conf", "best_area_ratio"
        ])
        writer.writeheader()
        if per_frame_rows:
            writer.writerows(per_frame_rows)

    return summary


def main():
    ensure_dir(OUT_DIR)
    model = YOLO(MODEL_PATH)

    all_summary = []

    for video_path, video_type in VIDEO_LIST:
        if not os.path.exists(video_path):
            print(f"[跳过] 视频不存在：{video_path}")
            continue

        print(f"\n=== 处理视频：{video_path} ===")

        # baseline
        base_out = os.path.join(OUT_DIR, "baseline")
        summary_base = run_one_mode(model, video_path, video_type, "baseline", RULE, base_out)
        all_summary.append(summary_base)
        print("[完成] baseline")

        # final
        final_out = os.path.join(OUT_DIR, "final")
        summary_final = run_one_mode(model, video_path, video_type, "final", RULE, final_out)
        all_summary.append(summary_final)
        print("[完成] final")

    # 汇总总表
    summary_csv = os.path.join(OUT_DIR, "summary.csv")
    if all_summary:
        with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_summary[0].keys()))
            writer.writeheader()
            writer.writerows(all_summary)

    print("\n全部完成。输出目录：")
    print(OUT_DIR)
    print("你重点看：summary.csv 和 screenshots/")

if __name__ == "__main__":
    main()
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ultralytics import YOLO

# =========================
# 你只需要改这里
# =========================
MODEL_PATH = r"runs\detect\yolov8m_150_no_copy_paste\weights\best.pt"  # 你的 best.pt
VIDEO_PATH = r"E:\User\ultralytics-8.3.241\video\vedio_2.mp4"  # 你的视频
OUT_ROOT = r"runs\system_eval"  # 输出目录（会自动创建）

TARGET_CLASS_ID = None  # None=任何类别都算；如果只关心 knife 类，一般就是 0


# =========================
# 评估方案（系统规则）列表：你可以按需增减
# =========================
# 建议：先跑 4~8 个方案够写论文
# 加“目标面积比”过滤：抑制远处小框噪声
# # 更严格
RULE_SETS = [
    # name, conf_th, min_hits, cooldown, min_area_ratio
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
    detected: int  # 0/1：本帧是否满足有效检测条件
    hit_count: int  # 连续命中计数（本帧处理后）
    triggered: int  # 0/1：本帧是否触发“预警事件”
    best_conf: float  # 本帧最强置信度（满足类筛选后取最大；没有则 0）
    best_area_ratio: float  # 与 best_conf 对应的 area_ratio；没有则 0
    n_boxes: int  # 本帧框数量（类筛选后）
    max_conf_any: float  # 本帧所有框最大 conf（类筛选后）
    mean_conf: float  # 本帧平均 conf（类筛选后；没有则 0）


@dataclass
class EventRecord:
    rule_name: str
    event_id: int
    start_frame: int
    end_frame: int
    start_t: float
    end_t: float
    duration_sec: float
    max_hit_in_event: int
    max_conf_in_event: float
    mean_conf_in_event: float


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


def read_video_info(cap: cv2.VideoCapture) -> tuple[int, int, float, int]:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return w, h, fps, frame_count


def extract_boxes(r, target_cls: int | None) -> list[tuple[float, int, tuple[float, float, float, float]]]:
    """返回 [(conf, cls, (x1,y1,x2,y2)), ...]，已按 target_cls 筛选."""
    out = []
    if r.boxes is None or len(r.boxes) == 0:
        return out

    confs = r.boxes.conf
    clss = r.boxes.cls
    xyxy = r.boxes.xyxy

    for conf, cls, box in zip(confs, clss, xyxy):
        c = safe_float(conf)
        k = int(cls)
        if target_cls is not None and k != target_cls:
            continue
        x1, y1, x2, y2 = map(float, box)
        out.append((c, k, (x1, y1, x2, y2)))
    return out


def calc_area_ratio(box, frame_area: float) -> float:
    x1, y1, x2, y2 = box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area / frame_area if frame_area > 0 else 0.0


# =========================
# 核心：跑一个规则集
# =========================
def run_one_rule(
    model: YOLO,
    video_path: str,
    rule: RuleConfig,
    out_dir: str,
    imgsz: int = 640,
    iou: float = 0.7,
    device: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """返回： df_frames: 每帧记录 df_events: 事件记录 summary: 汇总字典.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    W, H, fps, frame_count = read_video_info(cap)
    frame_area = float(W * H)
    duration_sec = frame_count / fps if fps > 0 else 0.0

    hit_count = 0
    last_trigger_t = -1e9
    frames: list[FrameRecord] = []

    # 事件统计：把一次“连续命中”段落当做 event（更方便论文）
    events: list[EventRecord] = []
    in_event = False
    event_id = 0
    event_start_frame = 0
    event_start_t = 0.0
    event_max_hit = 0
    event_conf_list: list[float] = []
    event_max_conf = 0.0

    # 计时：推理性能
    t0_wall = time.time()
    infer_time_sum = 0.0
    n_infer = 0

    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_sec = frame_idx / fps if fps > 0 else float(frame_idx)

        # 推理
        t_infer0 = time.time()
        # 用 predict 比 track 更稳定可控（你这里做的是检测+规则，不需要追踪ID）
        res = model.predict(
            frame,
            imgsz=imgsz,
            conf=0.001,  # 先尽量多出框，后面用 rule.conf_th 再过滤
            iou=iou,
            device=device,
            verbose=False,
        )
        t_infer1 = time.time()
        infer_time_sum += t_infer1 - t_infer0
        n_infer += 1

        r = res[0]
        boxes = extract_boxes(r, TARGET_CLASS_ID)

        n_boxes = len(boxes)
        conf_list = [b[0] for b in boxes] if n_boxes > 0 else []
        max_conf_any = max(conf_list) if conf_list else 0.0
        mean_conf = float(np.mean(conf_list)) if conf_list else 0.0

        # 选出“满足规则”的最佳框（conf最高且通过面积比）
        best_conf = 0.0
        best_area_ratio = 0.0
        detected = 0

        if boxes:
            # 按 conf 从高到低
            boxes_sorted = sorted(boxes, key=lambda x: x[0], reverse=True)
            for conf, cls, box in boxes_sorted:
                area_ratio = calc_area_ratio(box, frame_area)
                if conf >= rule.conf_th and area_ratio >= rule.min_area_ratio:
                    detected = 1
                    best_conf = conf
                    best_area_ratio = area_ratio
                    break

        # 连续命中逻辑
        if detected:
            hit_count += 1
        else:
            hit_count = 0

        # 触发逻辑（系统规则）
        triggered = 0
        if hit_count >= rule.min_hits and (t_sec - last_trigger_t) >= rule.cooldown_sec:
            triggered = 1
            last_trigger_t = t_sec
            hit_count = 0  # 触发后清零，避免连续重复触发

        frames.append(
            FrameRecord(
                rule_name=rule.name,
                frame_idx=frame_idx,
                t_sec=t_sec,
                detected=detected,
                hit_count=hit_count,
                triggered=triggered,
                best_conf=best_conf,
                best_area_ratio=best_area_ratio,
                n_boxes=n_boxes,
                max_conf_any=max_conf_any,
                mean_conf=mean_conf,
            )
        )

        # 事件段落：以 detected=1 的连续段为事件（注意：触发 triggered 可能在事件中出现）
        if detected and not in_event:
            in_event = True
            event_id += 1
            event_start_frame = frame_idx
            event_start_t = t_sec
            event_max_hit = 1
            event_conf_list = [best_conf] if best_conf > 0 else []
            event_max_conf = best_conf
        elif detected and in_event:
            event_max_hit = max(event_max_hit, hit_count if hit_count > 0 else event_max_hit)
            if best_conf > 0:
                event_conf_list.append(best_conf)
                event_max_conf = max(event_max_conf, best_conf)
        elif (not detected) and in_event:
            # 事件结束
            in_event = False
            end_frame = frame_idx - 1
            end_t = end_frame / fps if fps > 0 else float(end_frame)
            dur = max(0.0, end_t - event_start_t)
            mean_event_conf = float(np.mean(event_conf_list)) if event_conf_list else 0.0
            events.append(
                EventRecord(
                    rule_name=rule.name,
                    event_id=event_id,
                    start_frame=event_start_frame,
                    end_frame=end_frame,
                    start_t=event_start_t,
                    end_t=end_t,
                    duration_sec=dur,
                    max_hit_in_event=event_max_hit,
                    max_conf_in_event=event_max_conf,
                    mean_conf_in_event=mean_event_conf,
                )
            )

    # 结尾仍在事件中
    if in_event:
        end_frame = frame_idx
        end_t = end_frame / fps if fps > 0 else float(end_frame)
        dur = max(0.0, end_t - event_start_t)
        mean_event_conf = float(np.mean(event_conf_list)) if event_conf_list else 0.0
        events.append(
            EventRecord(
                rule_name=rule.name,
                event_id=event_id,
                start_frame=event_start_frame,
                end_frame=end_frame,
                start_t=event_start_t,
                end_t=end_t,
                duration_sec=dur,
                max_hit_in_event=event_max_hit,
                max_conf_in_event=event_max_conf,
                mean_conf_in_event=mean_event_conf,
            )
        )

    cap.release()
    t1_wall = time.time()

    # DataFrame
    df_frames = pd.DataFrame([asdict(x) for x in frames])
    df_events = pd.DataFrame([asdict(x) for x in events])

    # 汇总指标（论文常用）
    n_frames = len(df_frames)
    n_detected_frames = int(df_frames["detected"].sum()) if n_frames else 0
    n_triggers = int(df_frames["triggered"].sum()) if n_frames else 0

    triggers_per_min = (n_triggers / (duration_sec / 60.0)) if duration_sec > 0 else 0.0
    detected_ratio = (n_detected_frames / n_frames) if n_frames > 0 else 0.0

    # 平均推理耗时 / FPS
    avg_infer_ms = (infer_time_sum / n_infer * 1000.0) if n_infer > 0 else 0.0
    eff_fps = (n_infer / infer_time_sum) if infer_time_sum > 0 else 0.0

    # 触发间隔（秒）
    trigger_times = df_frames.loc[df_frames["triggered"] == 1, "t_sec"].to_numpy()
    if len(trigger_times) >= 2:
        trigger_intervals = np.diff(trigger_times)
        mean_trigger_interval = float(np.mean(trigger_intervals))
        min_trigger_interval = float(np.min(trigger_intervals))
    else:
        mean_trigger_interval = 0.0
        min_trigger_interval = 0.0

    # 最大连续命中（从 hit_count 看不到“触发前的峰值”完全准确，但可从事件里取）
    max_hit_event = int(df_events["max_hit_in_event"].max()) if len(df_events) else 0

    # 置信度统计（只看 best_conf>0 的帧）
    best_conf_valid = df_frames.loc[df_frames["best_conf"] > 0, "best_conf"].to_numpy()
    if len(best_conf_valid) > 0:
        mean_best_conf = float(np.mean(best_conf_valid))
        p50_best_conf = float(np.percentile(best_conf_valid, 50))
        p90_best_conf = float(np.percentile(best_conf_valid, 90))
    else:
        mean_best_conf = p50_best_conf = p90_best_conf = 0.0

    # 保存文件
    ensure_dir(out_dir)
    df_frames.to_csv(os.path.join(out_dir, "per_frame.csv"), index=False, encoding="utf-8-sig")
    df_events.to_csv(os.path.join(out_dir, "events.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "rule_name": rule.name,
        "video_path": video_path,
        "width": W,
        "height": H,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "conf_th": rule.conf_th,
        "min_hits": rule.min_hits,
        "cooldown_sec": rule.cooldown_sec,
        "min_area_ratio": rule.min_area_ratio,
        "n_frames": n_frames,
        "detected_frames": n_detected_frames,
        "detected_ratio": detected_ratio,
        "trigger_count": n_triggers,
        "triggers_per_min": triggers_per_min,
        "mean_trigger_interval_sec": mean_trigger_interval,
        "min_trigger_interval_sec": min_trigger_interval,
        "max_hit_in_event": max_hit_event,
        "avg_infer_ms": avg_infer_ms,
        "effective_fps": eff_fps,
        "mean_best_conf": mean_best_conf,
        "p50_best_conf": p50_best_conf,
        "p90_best_conf": p90_best_conf,
        "wall_time_sec": (t1_wall - t0_wall),
    }

    # 画图（便于论文）
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(plots_dir)

    # 1) detected 序列
    plt.figure()
    plt.plot(df_frames["t_sec"], df_frames["detected"])
    plt.xlabel("Time (s)")
    plt.ylabel("Detected (0/1)")
    plt.title(f"Detected Timeline - {rule.name}")
    plt.savefig(os.path.join(plots_dir, "detected_timeline.png"), dpi=200)
    plt.close()

    # 2) best_conf 分布
    plt.figure()
    if len(best_conf_valid) > 0:
        plt.hist(best_conf_valid, bins=30)
    plt.xlabel("best_conf")
    plt.ylabel("count")
    plt.title(f"best_conf Histogram - {rule.name}")
    plt.savefig(os.path.join(plots_dir, "best_conf_hist.png"), dpi=200)
    plt.close()

    # 3) trigger 时间点
    plt.figure()
    plt.plot(df_frames["t_sec"], df_frames["triggered"])
    plt.xlabel("Time (s)")
    plt.ylabel("Triggered (0/1)")
    plt.title(f"Trigger Timeline - {rule.name}")
    plt.savefig(os.path.join(plots_dir, "trigger_timeline.png"), dpi=200)
    plt.close()

    # 4) events duration 分布
    plt.figure()
    if len(df_events) > 0:
        plt.hist(df_events["duration_sec"], bins=20)
    plt.xlabel("event duration (s)")
    plt.ylabel("count")
    plt.title(f"Event Duration Histogram - {rule.name}")
    plt.savefig(os.path.join(plots_dir, "event_duration_hist.png"), dpi=200)
    plt.close()

    # summary 写入
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "summary.csv"), index=False, encoding="utf-8-sig")

    return df_frames, df_events, summary


# =========================
# 主程序：一次性跑完全部规则集
# =========================
def main():
    ensure_dir(OUT_ROOT)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(OUT_ROOT, f"video_eval_{timestamp}")
    ensure_dir(run_root)

    # 保存本次配置（方便复现）
    config_path = os.path.join(run_root, "run_config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"MODEL_PATH={MODEL_PATH}\n")
        f.write(f"VIDEO_PATH={VIDEO_PATH}\n")
        f.write(f"TARGET_CLASS_ID={TARGET_CLASS_ID}\n")
        for rs in RULE_SETS:
            f.write(str(rs) + "\n")

    model = YOLO(MODEL_PATH)

    all_summaries = []
    for name, conf_th, min_hits, cooldown, min_area_ratio in RULE_SETS:
        rule = RuleConfig(name, conf_th, min_hits, cooldown, min_area_ratio)
        out_dir = os.path.join(run_root, rule.name)
        print(f"\n=== Running: {rule.name} ===")
        _, _, summary = run_one_rule(
            model=model,
            video_path=VIDEO_PATH,
            rule=rule,
            out_dir=out_dir,
            imgsz=640,
            iou=0.7,
            device=0,  # 有GPU写0；没有GPU可改为 None 或 "cpu"
        )
        all_summaries.append(summary)

    df_all = pd.DataFrame(all_summaries)
    df_all.to_csv(os.path.join(run_root, "ALL_summary.csv"), index=False, encoding="utf-8-sig")

    # 生成一个“论文用对比表”友好的简表（你后面直接截图/贴表）
    cols = [
        "rule_name",
        "conf_th",
        "min_hits",
        "cooldown_sec",
        "min_area_ratio",
        "trigger_count",
        "triggers_per_min",
        "detected_ratio",
        "mean_trigger_interval_sec",
        "avg_infer_ms",
        "effective_fps",
        "mean_best_conf",
        "p50_best_conf",
        "p90_best_conf",
    ]
    df_comp = df_all[cols].copy()
    df_comp.to_csv(os.path.join(run_root, "PAPER_table.csv"), index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Outputs saved to: {run_root}")


if __name__ == "__main__":
    main()

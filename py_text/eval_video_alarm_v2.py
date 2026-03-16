# eval_video_alarm_v2.py
# 两种使用方式并存：
# 方式A：直接改本文件顶部 DEFAULT_* 配置，然后直接 python eval_video_alarm_v2.py
# 方式B：命令行传参覆盖默认值（与之前一致）

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO

# =========================================================
# 方式A：手动填写默认配置（不想打命令行就改这里）
# =========================================================
DEFAULT_VIDEO_PATH = r"E:\User\ultralytics-8.3.241\video\vedio_2.mp4"
DEFAULT_MODEL_PATH = r"E:\User\ultralytics-8.3.241\runs\detect\yolov8m_150_no_copy_paste\weights\best.pt"
DEFAULT_OUT_DIR = r"E:\User\ultralytics-8.3.241\runs\eval_video\video_2"

DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = "0"  # GPU填 "0"，CPU填 "cpu"

# 规则默认值（论文友好）
DEFAULT_CONF = 0.70
DEFAULT_IOU = 0.50
DEFAULT_HITS = 3
DEFAULT_MISS = 2
DEFAULT_MIN_EVENT = 0.30
DEFAULT_COOLDOWN = 2.0
DEFAULT_MIN_AREA = 0.0
DEFAULT_CLASS_ID = -1  # -1 表示所有类别
DEFAULT_MAX_DET = 50
DEFAULT_SAVE_PLOTS = False  # True 则保存png图（不显示窗口）


# -----------------------------
# 规则参数（论文友好、可解释）
# -----------------------------
@dataclass
class RuleConfig:
    conf_th: float = DEFAULT_CONF
    iou_th: float = DEFAULT_IOU
    hits_required: int = DEFAULT_HITS
    miss_tolerance: int = DEFAULT_MISS
    min_event_sec: float = DEFAULT_MIN_EVENT
    cooldown_sec: float = DEFAULT_COOLDOWN
    min_area_ratio: float = DEFAULT_MIN_AREA
    class_id: int | None = None
    max_det: int = DEFAULT_MAX_DET


# -----------------------------
# 单帧记录
# -----------------------------
@dataclass
class FrameRecord:
    video: str
    frame: int
    t_sec: float
    fps_video: float
    infer_ms: float

    best_conf: float
    best_area_ratio: float
    best_cls: int

    raw_hit: int
    stable: int
    triggered: int


# -----------------------------
# 事件记录（按stable片段聚合）
# -----------------------------
@dataclass
class EventRecord:
    video: str
    event_id: int
    start_frame: int
    end_frame: int
    start_t: float
    end_t: float
    duration_sec: float
    frames: int
    max_conf: float
    mean_conf: float


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    arr = np.array(values, dtype=np.float32)
    return float(np.percentile(arr, q))


def pick_best_det(
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    clss: np.ndarray,
    img_area: float,
    min_area_ratio: float,
    class_id: int | None,
) -> tuple[int, float, float, int]:
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return 0, 0.0, 0.0, -1

    best_idx = -1
    best_conf = -1.0
    best_area_ratio = 0.0
    best_cls = -1

    for i in range(len(boxes_xyxy)):
        c = float(confs[i])
        cls_i = int(clss[i])

        if class_id is not None and cls_i != class_id:
            continue

        x1, y1, x2, y2 = boxes_xyxy[i].astype(float).tolist()
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area_ratio = (bw * bh) / max(1.0, img_area)

        if area_ratio < min_area_ratio:
            continue

        if c > best_conf:
            best_conf = c
            best_idx = i
            best_area_ratio = float(area_ratio)
            best_cls = cls_i

    if best_idx < 0:
        return 0, 0.0, 0.0, -1

    return 1, float(best_conf), float(best_area_ratio), int(best_cls)


def run_eval(
    video_path: str,
    model_path: str,
    out_dir: str,
    rules: RuleConfig,
    imgsz: int = 640,
    device: str = "0",
    save_plots: bool = False,
):
    ensure_dir(out_dir)

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps_video = float(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0.0

    hit_streak = 0
    miss_streak_in_stable = 0
    stable_state = 0

    current_event_frames: list[FrameRecord] = []
    events: list[EventRecord] = []
    event_id = 0

    last_trigger_time = -1e9

    frames_out: list[FrameRecord] = []
    infer_times: list[float] = []

    frame_idx = -1
    start_wall = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        h, w = frame.shape[:2]
        img_area = float(h * w)
        t_sec = frame_idx / fps_video if fps_video > 0 else float(frame_idx)

        # 推理
        t0 = time.time()
        results = model.predict(
            source=frame,
            imgsz=imgsz,
            conf=rules.conf_th,
            iou=rules.iou_th,
            device=device,
            verbose=False,
            max_det=rules.max_det,
        )
        infer_ms = (time.time() - t0) * 1000.0
        infer_times.append(infer_ms)

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            raw_hit, best_conf, best_area_ratio, best_cls = 0, 0.0, 0.0, -1
        else:
            boxes_xyxy = r0.boxes.xyxy.detach().cpu().numpy()
            confs = r0.boxes.conf.detach().cpu().numpy()
            clss = r0.boxes.cls.detach().cpu().numpy()
            raw_hit, best_conf, best_area_ratio, best_cls = pick_best_det(
                boxes_xyxy,
                confs,
                clss,
                img_area,
                rules.min_area_ratio,
                rules.class_id,
            )

        # 稳定判定：连续hits进入；stable期间允许miss_tolerance
        if raw_hit == 1:
            hit_streak += 1
            miss_streak_in_stable = 0
        else:
            hit_streak = 0
            if stable_state == 1:
                miss_streak_in_stable += 1

        if stable_state == 0:
            if hit_streak >= rules.hits_required:
                stable_state = 1
                miss_streak_in_stable = 0
        else:
            if miss_streak_in_stable > rules.miss_tolerance:
                stable_state = 0
                miss_streak_in_stable = 0
                hit_streak = 0

        # 事件聚合（stable片段）
        triggered = 0
        if stable_state == 1:
            current_event_frames.append(
                FrameRecord(
                    video=video_name,
                    frame=frame_idx,
                    t_sec=t_sec,
                    fps_video=fps_video,
                    infer_ms=infer_ms,
                    best_conf=best_conf,
                    best_area_ratio=best_area_ratio,
                    best_cls=best_cls,
                    raw_hit=raw_hit,
                    stable=1,
                    triggered=0,
                )
            )
        else:
            if current_event_frames:
                start_t = current_event_frames[0].t_sec
                end_t = current_event_frames[-1].t_sec
                duration = max(0.0, end_t - start_t)

                if duration >= rules.min_event_sec:
                    confs_evt = [fr.best_conf for fr in current_event_frames]
                    events.append(
                        EventRecord(
                            video=video_name,
                            event_id=event_id,
                            start_frame=current_event_frames[0].frame,
                            end_frame=current_event_frames[-1].frame,
                            start_t=start_t,
                            end_t=end_t,
                            duration_sec=duration,
                            frames=len(current_event_frames),
                            max_conf=float(np.max(confs_evt)),
                            mean_conf=float(np.mean(confs_evt)),
                        )
                    )
                    event_id += 1

                current_event_frames = []

        # 触发：stable且不在冷却期 -> 触发一次
        if stable_state == 1:
            if (t_sec - last_trigger_time) >= rules.cooldown_sec:
                triggered = 1
                last_trigger_time = t_sec
                if current_event_frames:
                    current_event_frames[-1].triggered = 1

        frames_out.append(
            FrameRecord(
                video=video_name,
                frame=frame_idx,
                t_sec=t_sec,
                fps_video=fps_video,
                infer_ms=infer_ms,
                best_conf=best_conf,
                best_area_ratio=best_area_ratio,
                best_cls=best_cls,
                raw_hit=raw_hit,
                stable=stable_state,
                triggered=triggered,
            )
        )

    # 补结算最后事件
    if current_event_frames:
        start_t = current_event_frames[0].t_sec
        end_t = current_event_frames[-1].t_sec
        duration = max(0.0, end_t - start_t)

        if duration >= rules.min_event_sec:
            confs_evt = [fr.best_conf for fr in current_event_frames]
            events.append(
                EventRecord(
                    video=video_name,
                    event_id=event_id,
                    start_frame=current_event_frames[0].frame,
                    end_frame=current_event_frames[-1].frame,
                    start_t=start_t,
                    end_t=end_t,
                    duration_sec=duration,
                    frames=len(current_event_frames),
                    max_conf=float(np.max(confs_evt)),
                    mean_conf=float(np.mean(confs_evt)),
                )
            )
            event_id += 1

    cap.release()
    elapsed_wall = time.time() - start_wall

    # 输出CSV
    per_frame_path = os.path.join(out_dir, f"{video_name}_per_frame.csv")
    events_path = os.path.join(out_dir, f"{video_name}_events.csv")
    summary_path = os.path.join(out_dir, f"{video_name}_summary.csv")

    df_pf = pd.DataFrame([asdict(x) for x in frames_out])
    df_pf.to_csv(per_frame_path, index=False, encoding="utf-8-sig")

    df_ev = pd.DataFrame([asdict(x) for x in events])
    df_ev.to_csv(events_path, index=False, encoding="utf-8-sig")

    duration_sec = float(df_pf["t_sec"].max()) if len(df_pf) else 0.0
    frames_n = len(df_pf)
    stable_rate = float(df_pf["stable"].mean()) if frames_n else 0.0
    raw_rate = float(df_pf["raw_hit"].mean()) if frames_n else 0.0
    trigger_count = int(df_pf["triggered"].sum()) if frames_n else 0

    event_count = len(df_ev)
    event_mean = float(df_ev["duration_sec"].mean()) if event_count else 0.0
    event_max = float(df_ev["duration_sec"].max()) if event_count else 0.0
    event_p90 = float(np.percentile(df_ev["duration_sec"].values, 90)) if event_count else 0.0

    avg_infer = float(np.mean(infer_times)) if infer_times else float("nan")
    p95_infer = percentile(infer_times, 95)
    p99_infer = percentile(infer_times, 99)

    proc_fps = (frames_n / elapsed_wall) if elapsed_wall > 0 else float("nan")

    summary = {
        "video": video_name,
        "video_path": video_path,
        "model_path": model_path,
        "duration_sec": duration_sec,
        "frames": frames_n,
        "fps_video": fps_video,
        "proc_fps": proc_fps,
        "conf_th": rules.conf_th,
        "iou_th": rules.iou_th,
        "hits_required": rules.hits_required,
        "miss_tolerance": rules.miss_tolerance,
        "min_event_sec": rules.min_event_sec,
        "cooldown_sec": rules.cooldown_sec,
        "min_area_ratio": rules.min_area_ratio,
        "class_id": rules.class_id if rules.class_id is not None else -1,
        "max_det": rules.max_det,
        "raw_detect_rate": raw_rate,
        "stable_detect_rate": stable_rate,
        "trigger_count": trigger_count,
        "event_count": event_count,
        "event_mean_sec": event_mean,
        "event_p90_sec": event_p90,
        "event_max_sec": event_max,
        "avg_infer_ms": avg_infer,
        "p95_infer_ms": p95_infer,
        "p99_infer_ms": p99_infer,
    }

    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 可选：保存图（不显示）
    if save_plots:
        import matplotlib.pyplot as plt

        hit_confs = df_pf.loc[df_pf["raw_hit"] == 1, "best_conf"].values
        plt.figure()
        plt.hist(hit_confs, bins=30)
        plt.xlabel("best_conf")
        plt.ylabel("count")
        plt.title(f"best_conf Histogram - {video_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{video_name}_best_conf_hist.png"))
        plt.close()

        plt.figure()
        plt.plot(df_pf["t_sec"].values, df_pf["raw_hit"].values)
        plt.xlabel("Time (s)")
        plt.ylabel("raw_hit (0/1)")
        plt.title(f"Detected Timeline - {video_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{video_name}_detected_timeline.png"))
        plt.close()

        if event_count > 0:
            plt.figure()
            plt.hist(df_ev["duration_sec"].values, bins=30)
            plt.xlabel("event duration (s)")
            plt.ylabel("count")
            plt.title(f"Event Duration Histogram - {video_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{video_name}_event_duration_hist.png"))
            plt.close()

        plt.figure()
        plt.plot(df_pf["t_sec"].values, df_pf["triggered"].values)
        plt.xlabel("Time (s)")
        plt.ylabel("triggered (0/1)")
        plt.title(f"Trigger Timeline - {video_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{video_name}_trigger_timeline.png"))
        plt.close()

    print("Saved:")
    print("  per_frame:", per_frame_path)
    print("  events   :", events_path)
    print("  summary  :", summary_path)
    print("Done.")


def parse_args():
    ap = argparse.ArgumentParser(description="Video eval (CSV only), supports defaults in file head.")
    ap.add_argument("--video", type=str, default=DEFAULT_VIDEO_PATH, help="video path")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="model .pt path")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT_DIR, help="output dir")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="0 or cpu")

    # rule params（命令行传参可覆盖默认）
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--iou", type=float, default=DEFAULT_IOU)
    ap.add_argument("--hits", type=int, default=DEFAULT_HITS)
    ap.add_argument("--miss", type=int, default=DEFAULT_MISS)
    ap.add_argument("--min_event", type=float, default=DEFAULT_MIN_EVENT)
    ap.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN)
    ap.add_argument("--min_area", type=float, default=DEFAULT_MIN_AREA)
    ap.add_argument("--class_id", type=int, default=DEFAULT_CLASS_ID, help="-1 means all classes")
    ap.add_argument("--max_det", type=int, default=DEFAULT_MAX_DET)

    ap.add_argument("--plots", action="store_true", default=DEFAULT_SAVE_PLOTS, help="save plots png (no show)")
    return ap.parse_args()


def main():
    args = parse_args()

    # 如果你完全不传参，就会使用文件顶部 DEFAULT_* 的配置
    # 如果传了参，会覆盖对应 DEFAULT_*
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"video not found: {args.video}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"model not found: {args.model}")

    class_id = None if args.class_id < 0 else int(args.class_id)

    rules = RuleConfig(
        conf_th=args.conf,
        iou_th=args.iou,
        hits_required=args.hits,
        miss_tolerance=args.miss,
        min_event_sec=args.min_event,
        cooldown_sec=args.cooldown,
        min_area_ratio=args.min_area,
        class_id=class_id,
        max_det=args.max_det,
    )

    run_eval(
        video_path=args.video,
        model_path=args.model,
        out_dir=args.out,
        rules=rules,
        imgsz=args.imgsz,
        device=args.device,
        save_plots=args.plots,
    )


if __name__ == "__main__":
    main()

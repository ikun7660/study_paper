import os
import time

import cv2
import pandas as pd

from ultralytics import YOLO

# =========================
# 你需要改的 3 个路径
# =========================
VIDEO_PATH = r"E:\User\ultralytics-8.3.241\video\vedio_2.mp4"
MODEL_PATH = r"runs\detect\yolov8m_150_no_copy_paste\weights\best.pt"
OUT_DIR = r"E:\User\ultralytics-8.3.241\video_eval_out"

# =========================
# 规则/阈值（你可以调）
# =========================
CONF_TH = 0.70  # 置信度阈值：先用你当前的
MIN_AREA_RATIO = 0.00  # 最小面积占比（box_area / frame_area），0 就是不限制
HITS_TO_TRIGGER = 1  # 连续命中多少帧才触发（你当前是 1）
COOLDOWN_SEC = 0.0  # 触发后冷却秒数（你当前 0）
IMG_SIZE = 640  # 推理尺寸
DEVICE = 0  # 有 GPU 就 0；没有就 "cpu"
CLASS_ID = None  # 只做 knife 单类可不设；若你想只看某类填 0

# =========================
# 可视化开关
# =========================
SHOW_WINDOW = True  # 弹窗显示过程
SAVE_VIDEO = True  # 保存带叠加的视频
SAVE_CSV = True  # 保存 per_frame / events / summary
DRAW_ALL_RAW = True  # 画“模型原始输出框”（未过滤）
DRAW_AFTER_FILTER = True  # 画“过滤后框”（用于规则判断）
MAX_BOXES_DRAW = 30  # 防止太多框影响显示

# =========================
# UI 控制
# =========================
KEY_PAUSE = ord(" ")  # 空格暂停/继续
KEY_QUIT = ord("q")  # q 退出
KEY_STEP = ord("n")  # n 单步（暂停状态下）
KEY_TOGGLE_RAW = ord("r")  # r 显示/隐藏 raw 框
KEY_TOGGLE_FIL = ord("f")  # f 显示/隐藏 filter 框


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def box_area_xyxy(x1, y1, x2, y2):
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def draw_label(img, x, y, text, bg=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    if bg:
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 3), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def main():
    ensure_dir(OUT_DIR)

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_area = float(w * h)

    # 输出文件
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_video_path = os.path.join(
        OUT_DIR, f"debug_{ts}_conf{CONF_TH:.2f}_hits{HITS_TO_TRIGGER}_area{MIN_AREA_RATIO:.2f}_cd{COOLDOWN_SEC:.1f}.mp4"
    )
    out_perframe = os.path.join(OUT_DIR, f"per_frame_{ts}.csv")
    out_events = os.path.join(OUT_DIR, f"events_{ts}.csv")
    out_summary = os.path.join(OUT_DIR, f"summary_{ts}.csv")

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps if fps > 0 else 30, (w, h))

    # 记录数据
    per_rows = []
    events = []
    paused = False
    step_once = False

    # 规则状态
    hits = 0
    triggered = 0
    last_trigger_t = -1e9
    in_event = False
    event_start_f = None
    event_start_t = None

    show_raw = DRAW_ALL_RAW
    show_fil = DRAW_AFTER_FILTER

    frame_idx = 0
    time.time()

    while True:
        if SHOW_WINDOW and paused and not step_once:
            key = cv2.waitKey(30) & 0xFF
        else:
            ret, frame = cap.read()
            if not ret:
                break
            step_once = False

            # 推理
            infer_t0 = time.time()
            res = model.predict(
                source=frame,
                imgsz=IMG_SIZE,
                conf=0.001,  # 故意放低，让你看到“模型原始输出”到底有没有框
                iou=0.7,
                device=DEVICE,
                verbose=False,
            )[0]
            infer_ms = (time.time() - infer_t0) * 1000.0

            raw_boxes = []
            fil_boxes = []

            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), cf, c in zip(xyxy, confs, clss):
                    if CLASS_ID is not None and c != CLASS_ID:
                        continue
                    area_ratio = box_area_xyxy(x1, y1, x2, y2) / frame_area
                    raw_boxes.append((x1, y1, x2, y2, float(cf), int(c), float(area_ratio)))

                    # 过滤后用于规则判断
                    if cf >= CONF_TH and area_ratio >= MIN_AREA_RATIO:
                        fil_boxes.append((x1, y1, x2, y2, float(cf), int(c), float(area_ratio)))

            # 规则判断：这一帧是否“命中”
            detected = 1 if len(fil_boxes) > 0 else 0

            # 冷却
            t_sec = frame_idx / fps if fps > 0 else frame_idx / 30.0
            in_cooldown = (t_sec - last_trigger_t) < COOLDOWN_SEC

            if detected and (not in_cooldown):
                hits += 1
            else:
                hits = 0

            # 触发条件
            if (hits >= HITS_TO_TRIGGER) and (not in_cooldown):
                triggered = 1
                last_trigger_t = t_sec
            else:
                triggered = 0

            # 事件统计：把“检测事件”定义为 detected 连续为 1 的区间
            if detected and (not in_event):
                in_event = True
                event_start_f = frame_idx
                event_start_t = t_sec
            if (not detected) and in_event:
                in_event = False
                end_f = frame_idx - 1
                end_t = (end_f / fps) if fps > 0 else (end_f / 30.0)
                duration = max(0.0, end_t - event_start_t)
                events.append(
                    {
                        "event_id": len(events) + 1,
                        "start_frame": event_start_f,
                        "end_frame": end_f,
                        "start_time_s": event_start_t,
                        "end_time_s": end_t,
                        "duration_s": duration,
                    }
                )

            # 取最大置信度（raw / filter）
            best_raw = max([b[4] for b in raw_boxes], default=0.0)
            best_fil = max([b[4] for b in fil_boxes], default=0.0)

            # 记录 per-frame
            per_rows.append(
                {
                    "frame": frame_idx,
                    "time_s": t_sec,
                    "raw_count": len(raw_boxes),
                    "filter_count": len(fil_boxes),
                    "best_raw_conf": best_raw,
                    "best_filter_conf": best_fil,
                    "hits": hits,
                    "triggered": triggered,
                    "in_cooldown": int(in_cooldown),
                    "infer_ms": infer_ms,
                }
            )

            # ====== 画图层：让你肉眼判断“没检测”还是“被阈值过滤” ======
            vis = frame.copy()

            # RAW 框（模型原始输出）
            if show_raw:
                for i, (x1, y1, x2, y2, cf, c, ar) in enumerate(raw_boxes[:MAX_BOXES_DRAW]):
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (50, 50, 255), 2)
                    draw_label(vis, int(x1), int(y1), f"RAW c{c} conf={cf:.2f} area={ar:.4f}")

            # FILTER 后框（真正进入规则判断的）
            if show_fil:
                for i, (x1, y1, x2, y2, cf, c, ar) in enumerate(fil_boxes[:MAX_BOXES_DRAW]):
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    draw_label(vis, int(x1), int(y2) + 18, f"PASS conf>={CONF_TH:.2f} area>={MIN_AREA_RATIO:.2f}")

            # 顶部状态条
            draw_label(vis, 10, 25, f"frame {frame_idx}/{total}  t={t_sec:.2f}s  fps={fps:.2f}")
            draw_label(
                vis,
                10,
                50,
                f"raw={len(raw_boxes)}  filter={len(fil_boxes)}  best_raw={best_raw:.2f}  best_fil={best_fil:.2f}",
            )
            draw_label(
                vis, 10, 75, f"CONF_TH={CONF_TH:.2f}  MIN_AREA={MIN_AREA_RATIO:.2f}  HITS={hits}/{HITS_TO_TRIGGER}"
            )
            draw_label(vis, 10, 100, f"triggered={triggered}  cooldown={int(in_cooldown)}  infer={infer_ms:.1f}ms")

            if writer is not None:
                writer.write(vis)

            if SHOW_WINDOW:
                cv2.imshow("YOLO Video Debug (q quit / space pause / n step / r raw / f filter)", vis)
                key = cv2.waitKey(1) & 0xFF

                if key == KEY_QUIT:
                    break
                elif key == KEY_PAUSE:
                    paused = not paused
                elif key == KEY_STEP:
                    if paused:
                        step_once = True
                elif key == KEY_TOGGLE_RAW:
                    show_raw = not show_raw
                elif key == KEY_TOGGLE_FIL:
                    show_fil = not show_fil

            frame_idx += 1

    # 收尾：如果视频结束时仍在事件内，补上最后一个事件
    if in_event:
        end_f = frame_idx - 1
        end_t = (end_f / fps) if fps > 0 else (end_f / 30.0)
        duration = max(0.0, end_t - event_start_t)
        events.append(
            {
                "event_id": len(events) + 1,
                "start_frame": event_start_f,
                "end_frame": end_f,
                "start_time_s": event_start_t,
                "end_time_s": end_t,
                "duration_s": duration,
            }
        )

    cap.release()
    if writer is not None:
        writer.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    # 汇总指标
    df = pd.DataFrame(per_rows)
    df_events = pd.DataFrame(events)

    summary = {
        "video": VIDEO_PATH,
        "model": MODEL_PATH,
        "CONF_TH": CONF_TH,
        "MIN_AREA_RATIO": MIN_AREA_RATIO,
        "HITS_TO_TRIGGER": HITS_TO_TRIGGER,
        "COOLDOWN_SEC": COOLDOWN_SEC,
        "frames": int(df.shape[0]),
        "duration_s": float(df["time_s"].max() if len(df) else 0.0),
        "raw_any_frame_rate": float((df["raw_count"] > 0).mean() if len(df) else 0.0),
        "detected_frame_rate": float((df["filter_count"] > 0).mean() if len(df) else 0.0),
        "max_best_raw_conf": float(df["best_raw_conf"].max() if len(df) else 0.0),
        "max_best_filter_conf": float(df["best_filter_conf"].max() if len(df) else 0.0),
        "trigger_times": int(df["triggered"].sum() if len(df) else 0),
        "events": int(df_events.shape[0]),
        "avg_event_duration_s": float(df_events["duration_s"].mean() if len(df_events) else 0.0),
        "max_event_duration_s": float(df_events["duration_s"].max() if len(df_events) else 0.0),
        "avg_infer_ms": float(df["infer_ms"].mean() if len(df) else 0.0),
    }
    df_sum = pd.DataFrame([summary])

    if SAVE_CSV:
        df.to_csv(out_perframe, index=False, encoding="utf-8-sig")
        df_events.to_csv(out_events, index=False, encoding="utf-8-sig")
        df_sum.to_csv(out_summary, index=False, encoding="utf-8-sig")

    print("Done.")
    if SAVE_VIDEO:
        print("Saved video:", out_video_path)
    if SAVE_CSV:
        print("Saved per_frame:", out_perframe)
        print("Saved events   :", out_events)
        print("Saved summary  :", out_summary)


if __name__ == "__main__":
    main()

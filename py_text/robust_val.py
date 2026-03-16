# robust_val_all.py
# Ultralytics 8.3.241 兼容版：一次性评估 clean/noise/blur/dark
# 输出：
#   1) 每次val仍会在 runs/detect/val* 下生成曲线/混淆矩阵/示例图（save=True）
#   2) 额外生成一份总汇总CSV：runs/detect/robust_summary_YYYYMMDD_HHMMSS.csv

from ultralytics import YOLO
import csv
import os
from datetime import datetime

MODEL_PATH = r"runs\detect\yolov8m_150\weights\best.pt"

DATASETS = {
    "clean": r"datasets\knife_robust\clean\data.yaml",
    "noise": r"datasets\knife_robust\noise\data.yaml",
    "blur":  r"datasets\knife_robust\blur\data.yaml",
    "dark":  r"datasets\knife_robust\dark\data.yaml",
}

VAL_KWARGS = dict(
    imgsz=640,
    conf=0.001,
    iou=0.6,
    device=0,     # 没GPU就改成 "cpu" 或直接删掉
    workers=0,    # Windows 下避免多进程
    save=True,    # 保存曲线/混淆矩阵/示例图到 runs/detect/val*
)

def get_box_metrics(m):
    """
    兼容 Ultralytics 的指标对象，返回 (P, R, mAP50, mAP50-95)
    """
    # 常见字段：m.box.mp / m.box.mr / m.box.map50 / m.box.map
    box = getattr(m, "box", None)
    if box is None:
        return None, None, None, None

    def f(x):
        try:
            return float(x)
        except Exception:
            return None

    P = f(getattr(box, "mp", None))
    R = f(getattr(box, "mr", None))
    mAP50 = f(getattr(box, "map50", None))
    mAP5095 = f(getattr(box, "map", None))
    return P, R, mAP50, mAP5095

def get_save_dir(m):
    """
    尝试拿到本次 val 输出目录（runs/detect/val*）
    不同版本字段名可能不同，做容错。
    """
    for attr in ("save_dir", "dir"):
        if hasattr(m, attr):
            try:
                return str(getattr(m, attr))
            except Exception:
                pass
    return ""

def main():
    model = YOLO(MODEL_PATH)

    out_root = os.path.join("runs", "detect")
    os.makedirs(out_root, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(out_root, f"robust_summary_{ts}.csv")

    rows = []
    for name, yaml_path in DATASETS.items():
        print(f"\n==============================")
        print(f"Evaluating: {name}")
        print(f"data: {yaml_path}")
        print(f"==============================")

        m = model.val(data=yaml_path, **VAL_KWARGS)

        P, R, mAP50, mAP5095 = get_box_metrics(m)
        save_dir = get_save_dir(m)

        rows.append({
            "dataset": name,
            "P": P,
            "R": R,
            "mAP50": mAP50,
            "mAP50-95": mAP5095,
            "val_output_dir": save_dir
        })

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "P", "R", "mAP50", "mAP50-95", "val_output_dir"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nAll done.")
    print("Summary CSV saved to:")
    print(summary_path)
    print("\nPer-run plots/images are saved under runs/detect/val* (because save=True).")

if __name__ == "__main__":
    main()

import cv2
from ultralytics import YOLO
from pathlib import Path

# ======== 修改这里 ========
VIDEO_PATH = r"E:\User\ultralytics-8.3.241\video\vedio_2.mp4"
MODEL_PATH = r"E:\User\ultralytics-8.3.241\runs\detect\yolov8m_150_no_copy_paste\weights\best.pt"
OUT_DIR = r"./"
CONF_TH = 0.5     # 置信度阈值（传统YOLO用这个就够）
# =========================

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

frame_id = 0
save_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 推理
    results = model.predict(
        source=frame,
        conf=CONF_TH,
        iou=0.45,
        verbose=False
    )

    r = results[0]

    # 如果这一帧有检测结果
    if r.boxes is not None and len(r.boxes) > 0:
        img = r.plot()  # 原生YOLO画框
        save_path = f"{OUT_DIR}/frame_{frame_id:06d}.jpg"
        cv2.imwrite(save_path, img)
        save_id += 1

    frame_id += 1

cap.release()
print(f"完成：共保存 {save_id} 张检测帧")

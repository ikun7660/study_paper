import cv2

from ultralytics import YOLO

# ====================== 需你手动修改的配置（3处核心）=====================
MODEL_PATH = r"E:\User\ultralytics-8.3.241\yolov8m.pt"  # 你的训练好的YOLO模型路径（.pt格式）
VIDEO_PATH = r"E:\User\ultralytics-8.3.241\video\vedio_2.mp4"  # 你的测试视频路径（mp4/avi等OpenCV支持格式）
CONF_THRESHOLD = 0.5  # 检测置信度阈值，低于此值的框不显示（可调0.3-0.7）
# =========================================================================

# 可选配置：按需修改，不影响核心功能
DEVICE = "0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"  # 自动选GPU/CPU
WINDOW_NAME = "YOLO Real-Time Detection"  # OpenCV显示窗口名称
BOX_COLOR = (0, 255, 0)  # 检测框颜色（OpenCV是BGR，此为绿色）
TEXT_COLOR = (255, 255, 255)  # 标签文字颜色（白色）
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX  # 字体
TEXT_SIZE = 0.6  # 文字大小
BOX_THICKNESS = 2  # 检测框线宽
SAVE_RESULT = False  # 是否保存检测后的视频（False/True）
SAVE_PATH = "detect_result.mp4"  # 保存检测视频的路径（开启SAVE_RESULT时生效）

# 加载训练好的YOLO模型
model = YOLO(MODEL_PATH)
model.to(DEVICE)  # 模型推送到指定设备（GPU/CPU）
print(f"✅ 模型加载成功，使用设备：{DEVICE}")
print(f"✅ 检测置信度阈值：{CONF_THRESHOLD}")

# 打开测试视频
cap = cv2.VideoCapture(VIDEO_PATH)
# 检查视频是否成功打开
if not cap.isOpened():
    raise FileNotFoundError(f"❌ 无法打开测试视频，请检查路径：{VIDEO_PATH}")

# 获取视频基本信息（帧率、宽高），用于显示/保存
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"📹 视频信息：帧率={fps}，分辨率={frame_width}×{frame_height}，总帧数={total_frames}")

# 若开启保存，初始化视频写入器（OpenCV原生）
if SAVE_RESULT:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 视频编码格式（mp4推荐）
    out = cv2.VideoWriter(SAVE_PATH, fourcc, fps, (frame_width, frame_height))
    print(f"📽️  检测结果将保存至：{SAVE_PATH}")

# 实时检测主循环
frame_count = 0  # 帧计数器
print("\n🚀 开始实时检测，按【q/ESC】退出窗口...")
while cap.isOpened():
    # 逐帧读取视频
    ret, frame = cap.read()
    frame_count += 1

    # 读取完毕（视频结束）则退出循环
    if not ret:
        print(f"\n✅ 视频检测完成，共处理 {frame_count - 1} 帧")
        break

    # ---------------------- YOLO模型推理检测 ----------------------
    # stream=True：流式推理，节省内存，适合视频实时检测
    results = model(frame, conf=CONF_THRESHOLD, device=DEVICE, stream=True)

    # ---------------------- OpenCV绘制检测结果 ----------------------
    for res in results:
        # 获取检测结果：边界框、置信度、类别ID
        boxes = res.boxes.xyxy.cpu().numpy()  # 边界框坐标 (x1,y1,x2,y2)，转CPU+Numpy
        confs = res.boxes.conf.cpu().numpy()  # 置信度
        cls_ids = res.boxes.cls.cpu().numpy()  # 类别ID
        cls_names = res.names  # 类别名称（训练时的标签）

        # 遍历每个检测框，逐一生成绘制
        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = map(int, box)  # 坐标转整数（OpenCV绘制要求）
            cls_name = cls_names[int(cls_id)]  # 获取类别名称
            label = f"{cls_name} {conf:.2f}"  # 标签：类别+置信度（保留2位小数）

            # 1. 绘制检测框
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
            # 2. 绘制标签背景（避免文字与画面重叠，更清晰）
            label_size = cv2.getTextSize(label, TEXT_FONT, TEXT_SIZE, BOX_THICKNESS)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), BOX_COLOR, -1)
            # 3. 绘制标签文字
            cv2.putText(frame, label, (x1, y1 - 5), TEXT_FONT, TEXT_SIZE, TEXT_COLOR, BOX_THICKNESS)

    # ---------------------- OpenCV实时显示画面 ----------------------
    # 显示当前帧号和视频进度（可选，方便查看）
    progress = f"Frame: {frame_count}/{total_frames} | FPS: {fps}"
    cv2.putText(frame, progress, (10, 30), TEXT_FONT, 0.8, (0, 0, 255), 2)
    # 原生imshow显示，无任何重定向
    cv2.imshow(WINDOW_NAME, frame)

    # ---------------------- 保存检测结果（若开启） ----------------------
    if SAVE_RESULT:
        out.write(frame)

    # ---------------------- 退出按键检测 ----------------------
    # 按q/ESC键退出（OpenCV标准操作）
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        print(f"\n🛑 用户手动退出，共处理 {frame_count - 1} 帧")
        break

# 释放资源（OpenCV必须操作，避免内存泄漏）
cap.release()
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
if SAVE_RESULT:
    out.release()
    print(f"✅ 检测视频已保存：{SAVE_PATH}")

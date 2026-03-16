def run_one_rule(model, video_path, rule, out_dir):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    # 若视频无法打开，抛出异常
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    # 读取视频宽度、高度、帧率、总帧数
    W, H, fps, frame_count = read_video_info(cap)
    # 计算单帧面积
    frame_area = float(W * H)
    # 计算视频总时长（秒）
    duration_sec = frame_count / fps if fps > 0 else 0.0

    # 连续命中计数器（用于多帧稳定性判断）
    hit_count = 0
    # 上一次触发时间（用于冷却时间控制）
    last_trigger_t = -1e9
    # 存储每一帧的记录
    frames = []

    # 推理总耗时累计
    infer_time_sum = 0.0
    # 推理次数计数
    n_infer = 0

    # 当前帧索引初始化
    frame_idx = -1
    # 逐帧读取视频
    while True:
        ret, frame = cap.read()
        # 视频结束则退出循环
        if not ret:
            break

        # 更新帧索引
        frame_idx += 1
        # 当前时间（秒）
        t_sec = frame_idx / fps if fps > 0 else float(frame_idx)

        # 记录推理开始时间
        t0 = time.time()
        # 模型推理（低置信度阈值用于后续规则二次筛选）
        res = model.predict(frame, imgsz=640, conf=0.001, iou=0.7, verbose=False)
        # 累计推理耗时
        infer_time_sum += time.time() - t0
        n_infer += 1

        # 提取检测框并按类别筛选
        boxes = extract_boxes(res[0], TARGET_CLASS_ID)

        # 本帧检测框数量
        n_boxes = len(boxes)
        # 所有框的置信度列表
        conf_list = [b[0] for b in boxes] if boxes else []
        # 最大置信度
        max_conf_any = max(conf_list) if conf_list else 0.0
        # 平均置信度
        mean_conf = float(np.mean(conf_list)) if conf_list else 0.0

        # 满足规则的最佳置信度
        best_conf = 0.0
        # 满足规则的最佳面积占比
        best_area_ratio = 0.0
        # 是否检测到有效目标
        detected = 0

        # 若存在检测框，则按置信度排序
        if boxes:
            boxes_sorted = sorted(boxes, key=lambda x: x[0], reverse=True)
            # 遍历排序后的框，寻找第一个满足规则条件的框
            for conf, cls, box in boxes_sorted:
                area_ratio = calc_area_ratio(box, frame_area)
                # 同时满足置信度阈值与面积占比阈值
                if conf >= rule.conf_th and area_ratio >= rule.min_area_ratio:
                    detected = 1
                    best_conf = conf
                    best_area_ratio = area_ratio
                    break

        # 连续命中计数逻辑
        if detected:
            hit_count += 1
        else:
            hit_count = 0

        # 是否触发报警标志
        triggered = 0
        # 满足连续命中要求且超过冷却时间则触发
        if hit_count >= rule.min_hits and (t_sec - last_trigger_t) >= rule.cooldown_sec:
            triggered = 1
            last_trigger_t = t_sec
            # 触发后清零计数避免重复触发
            hit_count = 0

        # 记录当前帧信息
        frames.append(
            FrameRecord(
                rule.name,
                frame_idx,
                t_sec,
                detected,
                hit_count,
                triggered,
                best_conf,
                best_area_ratio,
                n_boxes,
                max_conf_any,
                mean_conf,
            )
        )

    # 释放视频资源
    cap.release()

    # 将逐帧记录转换为DataFrame
    df_frames = pd.DataFrame([asdict(x) for x in frames])
    # 确保输出目录存在
    ensure_dir(out_dir)
    # 保存逐帧CSV
    df_frames.to_csv(os.path.join(out_dir, "per_frame.csv"), index=False, encoding="utf-8-sig")

    # 总帧数
    n_frames = len(df_frames)
    # 检测到有效目标的帧数
    n_detected = int(df_frames["detected"].sum())
    # 触发次数
    n_triggers = int(df_frames["triggered"].sum())

    # 每分钟触发次数
    triggers_per_min = (n_triggers / (duration_sec / 60.0)) if duration_sec > 0 else 0.0
    # 检测帧占比
    detected_ratio = (n_detected / n_frames) if n_frames > 0 else 0.0

    # 平均推理耗时（毫秒）
    avg_infer_ms = (infer_time_sum / n_infer * 1000.0) if n_infer > 0 else 0.0
    # 有效帧率
    eff_fps = (n_infer / infer_time_sum) if infer_time_sum > 0 else 0.0

    # 汇总统计结果
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
        "effective_fps": eff_fps,
    }

    # 返回当前规则的汇总指标
    return summary


def main():
    # 确保输出根目录存在
    ensure_dir(OUT_ROOT)

    # 生成当前时间戳，用于区分不同实验批次
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # 构造本次运行的输出总目录
    run_root = os.path.join(OUT_ROOT, f"video_eval_{timestamp}")
    # 创建本次运行目录
    ensure_dir(run_root)

    # 加载YOLO模型权重
    model = YOLO(MODEL_PATH)

    # 用于存储所有规则组合的汇总结果
    all_summaries = []

    # 遍历预设的规则参数列表
    for name, conf_th, min_hits, cooldown, min_area_ratio in RULE_SETS:
        # 构造当前规则配置对象
        rule = RuleConfig(name, conf_th, min_hits, cooldown, min_area_ratio)
        # 为当前规则创建独立输出目录
        out_dir = os.path.join(run_root, rule.name)
        # 执行单条规则评估，返回汇总结果
        summary = run_one_rule(model, VIDEO_PATH, rule, out_dir)
        # 将当前规则的汇总结果加入列表
        all_summaries.append(summary)

    # 将所有规则汇总结果转换为DataFrame
    df_all = pd.DataFrame(all_summaries)
    # 保存规则对比总表
    df_all.to_csv(os.path.join(run_root, "ALL_summary.csv"), index=False, encoding="utf-8-sig")


# 程序入口判断，仅在直接运行该脚本时执行
if __name__ == "__main__":
    main()

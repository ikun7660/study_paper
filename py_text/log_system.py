# -*- coding: utf-8 -*-
"""
log_system.py
日志系统：运行日志窗口 + 异步文件写入 + 事件日志 + 汇总日志
"""

import os
import csv
import queue
from datetime import datetime

from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QPlainTextEdit, QLabel
)


# 后台写入线程：
# 负责把日志相关写入任务异步落盘，避免主线程直接写文件造成界面卡顿。
class LogWriterThread(QThread):

    # 初始化线程内部使用的任务队列和停止标记。
    # task_queue：主线程把待写入任务放进来，子线程统一消费。
    # _stop_flag：用于控制 run() 循环是否退出。
    def __init__(self, parent=None):
        super().__init__(parent)
        self.task_queue = queue.Queue()
        self._stop_flag = False

    # 投递一个日志写入任务到队列中。
    # task 一般是字典，里面包含：
    # - type：任务类型
    # - file_path：目标文件
    # - line / row：具体写入内容
    def enqueue(self, task: dict):
        self.task_queue.put(task)

    # 请求线程停止。
    # 这里只是设置停止标记还不够，因为 run() 可能正阻塞在 get()。
    # 所以额外投递一个 "__stop__" 哨兵任务，确保线程能及时醒来退出。
    def stop(self):
        self._stop_flag = True
        self.task_queue.put({"type": "__stop__"})

    # 确保目标文件的父目录存在。
    # 如果日志目录不存在，则先创建，防止 open() 写文件时报错。
    def _ensure_parent_dir(self, file_path: str):
        parent = os.path.dirname(file_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

    # 写入普通文本运行日志。
    # 每条日志都以一行文本追加到 .log 文件末尾。
    def _write_runtime_log(self, file_path: str, line: str):
        self._ensure_parent_dir(file_path)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # 写入事件明细 CSV。
    # 该文件主要记录命中、触发等关键事件，便于之后做实验分析或问题回溯。
    # 如果文件是第一次创建，则先写表头。
    def _write_event_csv(self, file_path: str, row: dict):
        self._ensure_parent_dir(file_path)
        file_exists = os.path.exists(file_path)

        # 事件级日志固定字段。
        # 这样无论调用方传入什么内容，最终导出的 CSV 列结构都是稳定的。
        headers = [
            "time_str", "event_type", "frame_id", "stage",
            "best_conf", "hits", "hits_required",
            "source_mode", "video_path", "cam_index"
        ]

        with open(file_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)

            # 只有新文件才写表头，避免重复写标题行。
            if not file_exists:
                writer.writeheader()

            # 按 headers 顺序输出，没提供的字段统一写空字符串。
            writer.writerow({k: row.get(k, "") for k in headers})

    # 写入运行汇总 CSV。
    # 该文件主要保存一次完整运行结束后的总体统计信息，
    # 例如总帧数、平均推理时间、最大推理时间、总命中数、总触发数等。
    def _write_summary_csv(self, file_path: str, row: dict):
        self._ensure_parent_dir(file_path)
        file_exists = os.path.exists(file_path)

        # 汇总级日志固定字段。
        headers = [
            "run_start_time", "run_end_time",
            "model_path", "source_mode", "video_path", "cam_index",
            "conf_th", "hits_required", "display_hits_required",
            "hit_window_sec", "min_area_ratio", "cooldown_sec",
            "imgsz", "iou", "device",
            "total_frames", "avg_infer_ms", "max_infer_ms",
            "total_hits", "total_triggers"
        ]

        with open(file_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)

            # 第一次写入时补表头。
            if not file_exists:
                writer.writeheader()

            # 统一字段顺序写入，缺失字段补空。
            writer.writerow({k: row.get(k, "") for k in headers})

    # 线程主循环。
    # 持续从任务队列中取出任务，根据 type 分发到不同写入函数。
    # 这里单独放在线程里做，能减少频繁写文件对主界面流畅度的影响。
    def run(self):
        while not self._stop_flag:
            try:
                # 设置 timeout，避免永久阻塞，便于周期性检查停止标记。
                task = self.task_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # 收到停止哨兵后直接退出循环。
            if task.get("type") == "__stop__":
                break

            try:
                task_type = task.get("type")

                # 普通运行日志
                if task_type == "runtime_log":
                    self._write_runtime_log(task["file_path"], task["line"])

                # 事件级 CSV
                elif task_type == "event_csv":
                    self._write_event_csv(task["file_path"], task["row"])

                # 汇总级 CSV
                elif task_type == "summary_csv":
                    self._write_summary_csv(task["file_path"], task["row"])

            except Exception:
                # 日志系统本身不应反向影响主程序运行，
                # 因此这里吞掉异常，避免日志写入失败导致主流程中断。
                pass


# 日志查看对话框：
# 在 GUI 中显示最近若干条运行日志，便于观察程序状态。
class LogViewerDialog(QDialog):

    # 初始化日志查看窗口。
    # 包含：
    # - 标题提示文本
    # - 只读日志文本框
    # - 清空显示按钮
    # - 关闭按钮
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("运行日志")
        self.resize(760, 480)

        # 顶部说明文字：提示界面里仅保留最近部分日志，文件日志不会被清空。
        self.label = QLabel("最近日志（界面仅保留最近若干条，日志文件不会清除）")

        # 文本显示区：使用 QPlainTextEdit 更适合大量纯文本输出。
        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)

        # 限制界面中最多保留的文本块数量，防止日志太多导致界面内存持续增长。
        self.text_edit.document().setMaximumBlockCount(500)

        # 控制按钮
        self.btn_clear = QPushButton("清空显示")
        self.btn_close = QPushButton("关闭")

        # 按钮水平布局
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.btn_close)

        # 主垂直布局
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.text_edit)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        # 清空按钮只清界面，不影响日志文件
        self.btn_clear.clicked.connect(self.text_edit.clear)

        # 关闭窗口
        self.btn_close.clicked.connect(self.close)

    # 向日志文本框追加一条日志。
    # 追加后自动把滚动条拉到底部，保证最新日志可见。
    def append_log(self, line: str):
        self.text_edit.appendPlainText(line)
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


# 日志管理器：
# 提供统一入口，供主界面或后台线程记录：
# - 普通文本日志
# - 事件日志 CSV
# - 汇总日志 CSV
class LogManager:

    # 初始化日志管理器。
    # log_dir：日志保存目录
    # max_gui_lines：界面上最多显示多少行（当前版本实际由 QPlainTextEdit 的 block 数控制）
    def __init__(self, log_dir: str, max_gui_lines: int = 500):
        self.log_dir = log_dir
        self.max_gui_lines = max_gui_lines

        # 日志查看窗口引用，可为空。
        self.viewer = None

        # 后台写入线程启动后持续可用。
        self.writer = LogWriterThread()
        self.writer.start()

        # 每次运行生成或更新的日志文件路径
        self.runtime_log_path = ""
        self.event_csv_path = ""
        self.summary_csv_path = ""

        # 当前运行批次的时间戳字符串，用于区分不同运行。
        self.current_run_stamp = ""

    # 绑定日志查看窗口。
    # 绑定后，log() 写日志时会同步推送到界面。
    def set_viewer(self, viewer: LogViewerDialog):
        self.viewer = viewer

    # 更新日志输出目录。
    # 当用户在 GUI 中手动选择新的日志目录时，会调用这里。
    def set_log_dir(self, log_dir: str):
        self.log_dir = log_dir

    # 为一次新运行生成新的日志文件名。
    # - runtime_时间戳.log：文本日志
    # - event_时间戳.csv：事件日志
    # - run_summary.csv：汇总日志（多次运行追加）
    def start_new_run(self):
        now = datetime.now()
        self.current_run_stamp = now.strftime("%Y%m%d_%H%M%S")

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        self.runtime_log_path = os.path.join(
            self.log_dir,
            f"runtime_{self.current_run_stamp}.log"
        )
        self.event_csv_path = os.path.join(
            self.log_dir,
            f"event_{self.current_run_stamp}.csv"
        )
        self.summary_csv_path = os.path.join(
            self.log_dir,
            "run_summary.csv"
        )

    # 停止后台写入线程并等待线程真正退出。
    # 这样可以避免程序结束时仍有未落盘日志。
    def stop(self):
        self.writer.stop()
        self.writer.wait()

    # 生成统一格式的当前时间字符串。
    # 所有普通文本日志都通过这里拿时间，格式保持一致。
    def _time_str(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 把日志同步显示到 GUI 窗口。
    # 如果当前没有绑定 viewer，则该步骤自动跳过。
    def _push_gui(self, line: str):
        if self.viewer is not None:
            self.viewer.append_log(line)

    # 写一条标准文本日志。
    # 会同时：
    # 1. 推送到 GUI 文本框
    # 2. 交给后台线程写入日志文件
    def log(self, level: str, message: str):
        line = f"[{self._time_str()}] [{level}] {message}"

        # 显示到界面
        self._push_gui(line)

        # 写入文件
        if self.runtime_log_path:
            self.writer.enqueue({
                "type": "runtime_log",
                "file_path": self.runtime_log_path,
                "line": line
            })

    # 记录 INFO 级日志。
    def info(self, message: str):
        self.log("INFO", message)

    # 记录 WARNING 级日志。
    def warning(self, message: str):
        self.log("WARNING", message)

    # 记录 ERROR 级日志。
    def error(self, message: str):
        self.log("ERROR", message)

    # 记录事件日志。
    # 通常用于记录命中、触发、阶段变化等关键过程数据。
    def event(self, row: dict):
        if self.event_csv_path:
            self.writer.enqueue({
                "type": "event_csv",
                "file_path": self.event_csv_path,
                "row": row
            })

    # 记录一次完整运行的汇总日志。
    # 常用于程序结束或停止后，将总统计结果写入 summary.csv。
    def summary(self, row: dict):
        if self.summary_csv_path:
            self.writer.enqueue({
                "type": "summary_csv",
                "file_path": self.summary_csv_path,
                "row": row
            })
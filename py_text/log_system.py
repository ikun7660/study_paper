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


class LogWriterThread(QThread):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.task_queue = queue.Queue()
        self._stop_flag = False

    def enqueue(self, task: dict):
        self.task_queue.put(task)

    def stop(self):
        self._stop_flag = True
        self.task_queue.put({"type": "__stop__"})

    def _ensure_parent_dir(self, file_path: str):
        parent = os.path.dirname(file_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

    def _write_runtime_log(self, file_path: str, line: str):
        self._ensure_parent_dir(file_path)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _write_event_csv(self, file_path: str, row: dict):
        self._ensure_parent_dir(file_path)
        file_exists = os.path.exists(file_path)
        headers = [
            "time_str", "event_type", "frame_id", "stage",
            "best_conf", "hits", "hits_required",
            "source_mode", "video_path", "cam_index"
        ]
        with open(file_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in headers})

    def _write_summary_csv(self, file_path: str, row: dict):
        self._ensure_parent_dir(file_path)
        file_exists = os.path.exists(file_path)
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
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in headers})

    def run(self):
        while not self._stop_flag:
            try:
                task = self.task_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if task.get("type") == "__stop__":
                break

            try:
                task_type = task.get("type")
                if task_type == "runtime_log":
                    self._write_runtime_log(task["file_path"], task["line"])
                elif task_type == "event_csv":
                    self._write_event_csv(task["file_path"], task["row"])
                elif task_type == "summary_csv":
                    self._write_summary_csv(task["file_path"], task["row"])
            except Exception:
                pass


class LogViewerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("运行日志")
        self.resize(760, 480)

        self.label = QLabel("最近日志（界面仅保留最近若干条，日志文件不会清除）")
        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.document().setMaximumBlockCount(500)

        self.btn_clear = QPushButton("清空显示")
        self.btn_close = QPushButton("关闭")

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.btn_close)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.text_edit)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.btn_clear.clicked.connect(self.text_edit.clear)
        self.btn_close.clicked.connect(self.close)

    def append_log(self, line: str):
        self.text_edit.appendPlainText(line)
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


class LogManager:
    def __init__(self, log_dir: str, max_gui_lines: int = 500):
        self.log_dir = log_dir
        self.max_gui_lines = max_gui_lines
        self.viewer = None
        self.writer = LogWriterThread()
        self.writer.start()

        self.runtime_log_path = ""
        self.event_csv_path = ""
        self.summary_csv_path = ""
        self.current_run_stamp = ""

    def set_viewer(self, viewer: LogViewerDialog):
        self.viewer = viewer

    def set_log_dir(self, log_dir: str):
        self.log_dir = log_dir

    def start_new_run(self):
        now = datetime.now()
        self.current_run_stamp = now.strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        self.runtime_log_path = os.path.join(self.log_dir, f"runtime_{self.current_run_stamp}.log")
        self.event_csv_path = os.path.join(self.log_dir, f"event_{self.current_run_stamp}.csv")
        self.summary_csv_path = os.path.join(self.log_dir, "run_summary.csv")

    def stop(self):
        self.writer.stop()
        self.writer.wait()

    def _time_str(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _push_gui(self, line: str):
        if self.viewer is not None:
            self.viewer.append_log(line)

    def log(self, level: str, message: str):
        line = f"[{self._time_str()}] [{level}] {message}"
        self._push_gui(line)
        if self.runtime_log_path:
            self.writer.enqueue({
                "type": "runtime_log",
                "file_path": self.runtime_log_path,
                "line": line
            })

    def info(self, message: str):
        self.log("INFO", message)

    def warning(self, message: str):
        self.log("WARNING", message)

    def error(self, message: str):
        self.log("ERROR", message)

    def event(self, row: dict):
        if self.event_csv_path:
            self.writer.enqueue({
                "type": "event_csv",
                "file_path": self.event_csv_path,
                "row": row
            })

    def summary(self, row: dict):
        if self.summary_csv_path:
            self.writer.enqueue({
                "type": "summary_csv",
                "file_path": self.summary_csv_path,
                "row": row
            })
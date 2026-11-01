"""Live video sampling: YOLO or YOLO + ConvLSTM, with automatic hardware selection."""

import queue
import sys
import threading
import time
import tkinter as tk
from collections import deque
from math import ceil, floor
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "raw/directml-runtime/onnxruntime").is_dir():
    sys.path.insert(0, str(ROOT / "raw/directml-runtime"))


def auto_device(path):
    import torch
    if Path(path).suffix.lower() == ".onnx":
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers and torch.cuda.is_available():
            return "cuda:0"
        return "dml" if "DmlExecutionProvider" in providers else "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def realtime_frames(capture, source_fps, duration, stop, started, clock=time.perf_counter):
    """Follow the source clock, dropping overdue samples instead of accumulating a backlog."""
    rate, previous, source_index = min(10, source_fps), -1, -1
    while not stop.is_set():
        due = max(previous + 1, floor((clock() - started) * rate + 1e-6))
        if stop.wait(max(0, started + due / rate - clock())):
            return
        due = max(due, floor((clock() - started) * rate + 1e-6))
        timestamp = due / rate
        if timestamp >= duration:
            return
        target = ceil(timestamp * source_fps - 1e-6)
        read_start = clock()
        while source_index < target:
            if stop.is_set() or not capture.grab():
                return
            source_index += 1
        ok, frame = capture.retrieve()
        if not ok:
            raise OSError("Không giải mã được video.")
        yield source_index / source_fps, frame, due - previous - 1, clock() - read_start
        previous = due


def predict_video(options, frames, events, stop):
    capture = None
    try:
        events.put(("status", "Đang nạp model và khởi động GPU…"))
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO
        try:
            from .demo_engine import TemporalPeople, action_classes, inspect_checkpoint, load_classifier
            from .onnx_models import Detector
        except ImportError:
            from demo_engine import TemporalPeople, action_classes, inspect_checkpoint, load_classifier
            from onnx_models import Detector

        if stop.is_set():
            return
        torch.set_num_threads(min(8, torch.get_num_threads()))
        if options["mode"] not in {"YOLO", "YOLO + ConvLSTM"}:
            raise ValueError("Chọn chế độ YOLO hoặc YOLO + ConvLSTM.")
        hybrid = options["mode"] == "YOLO + ConvLSTM"
        detector_path = Path(options["yolo"])
        if not detector_path.is_file() or inspect_checkpoint(detector_path)["kind"] != "yolo":
            raise ValueError("Ô YOLO cần file model YOLO drowning.")
        detector_device = auto_device(detector_path)
        detector = (Detector(detector_path, detector_device) if detector_path.suffix.lower() == ".onnx"
                    else YOLO(str(detector_path)))
        if detector.task != "detect":
            raise ValueError("Chọn YOLO detect hai lớp drowning/not_drowning.")
        classes = action_classes(detector.names)
        classifier, temporal_device = None, None
        if hybrid:
            main = Path(options["convlstm"])
            if not main.is_file() or inspect_checkpoint(main)["kind"] != "convlstm":
                raise ValueError("Ô ConvLSTM cần checkpoint ConvLSTM hoặc encoder.onnx.")
            temporal_device = auto_device(main)
            classifier = load_classifier(main, temporal_device if main.suffix.lower() == ".onnx"
                                         else torch.device(temporal_device))
        capture = cv2.VideoCapture(str(options["video"]))
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        width, height = (int(capture.get(key)) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if not capture.isOpened() or not 0 < source_fps < float("inf") or total <= 0 or min(width, height) <= 0:
            raise ValueError("Không đọc được thông số video.")
        if hybrid and source_fps < 10:
            raise ValueError("ConvLSTM train ở 10 FPS cần video nguồn ít nhất 10 FPS.")
        if stop.is_set():
            return
        people = TemporalPeople(classifier)
        predict = dict(conf=float(options.get("conf", .25)), iou=.5, classes=list(classes),
                       agnostic_nms=True, device=detector_device, verbose=False)
        # Initialize kernels before starting playback, including all temporal window sizes.
        detector.predict(np.zeros((height, width, 3), dtype=np.uint8), **predict)
        if classifier:
            c = classifier
            with torch.inference_mode(), torch.autocast(c.device.type, enabled=c.device.type == "cuda"):
                x = torch.zeros(1, 3, c.size, c.size, device=c.device).contiguous(memory_format=torch.channels_last)
                features = c.model.temporal.project_input(c.model.encoder(x))
                for length in sorted({min(8, c.length), min(16, c.length), c.length}):
                    if stop.is_set():
                        return
                    c.model.classify_features(features[:, None].repeat(1, length, 1, 1, 1), projected=True)
        cuda = detector_device.startswith("cuda") or (temporal_device or "").startswith("cuda")
        if cuda:
            torch.cuda.synchronize()
        backend = f"YOLO: {detector_device.upper()}"
        if classifier:
            backend += f" • ConvLSTM: {temporal_device.upper()} • ngưỡng P={classifier.threshold:.5f}"
        events.put(("backend", backend))
        events.put(("status", "Đang phát realtime • lấy tối đa 10 FPS • bỏ mẫu trễ khi máy không theo kịp"))
        duration, started = total / source_fps, time.perf_counter()
        timings, skipped = deque(maxlen=20), 0
        for timestamp, frame, dropped, read_time in realtime_frames(capture, source_fps, duration, stop, started):
            if stop.is_set():
                break
            skipped += dropped
            begin = time.perf_counter()
            result = detector.predict(frame, **predict)[0]
            if cuda:
                torch.cuda.synchronize()
            detector_ms = (time.perf_counter() - begin) * 1000
            annotated = people.update(frame, result, timestamp)
            if cuda:
                torch.cuda.synchronize()
            temporal_ms = (time.perf_counter() - begin) * 1000 - detector_ms
            scale = min(1, 1280 / width, 720 / height)
            if scale < 1:
                annotated = cv2.resize(annotated, (round(width * scale), round(height * scale)))
            image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            work = time.perf_counter() - begin + read_time
            timings.append(work)
            packet = dict(image=image, fps=len(timings) / sum(timings), skipped=skipped,
                          timestamp=timestamp, duration=duration, source_fps=source_fps,
                          produced=time.perf_counter(), started=started, people=len(result.boxes),
                          detector_ms=detector_ms, temporal_ms=temporal_ms,
                          predictions=people.predictions, target_fps=min(10, source_fps))
            if frames.full():
                try:
                    frames.get_nowait()
                except queue.Empty:
                    pass
            frames.put_nowait(packet)
        if not stop.is_set():
            events.put(("status", "Đã hết video. Có thể đổi file hoặc chạy lại."))
    except Exception as error:
        events.put(("error", str(error)))
    finally:
        if capture is not None:
            capture.release()
        events.put(("done", None))


class DemoApp:
    def __init__(self, root):
        self.root = root
        root.title("Drowning Detection Demo")
        root.geometry("1230x800")
        root.minsize(980, 780)
        self.frames, self.events = queue.Queue(maxsize=1), queue.SimpleQueue()
        self.stop_event, self.thread = threading.Event(), None
        self.finished = False
        self.display_times, self.photo = deque(maxlen=30), None
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", padding=7)
        outer = ttk.Frame(root, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Drowning Detection", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(outer, text="Realtime 1× • Tối đa 10 FPS • GPU tự động • PT / ONNX").pack(anchor="w", pady=(0, 12))
        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        sidebar = ttk.Frame(body, width=300)
        sidebar.pack(side="left", fill="y", padx=(0, 18))
        sidebar.pack_propagate(False)
        self.controls = []
        self.video, self.yolo, self.convlstm = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.file_field(sidebar, "Video", self.video, "demo", [("Video", "*.mp4 *.avi *.mkv *.mov *.webm"), ("All", "*.*")])
        types = [("Model", "*.pt *.pth *.onnx"), ("All", "*.*")]
        self.file_field(sidebar, "Model YOLO", self.yolo, "models", types)
        self.file_field(sidebar, "Model ConvLSTM", self.convlstm, "models", types)
        ttk.Label(sidebar, text="ONNX ConvLSTM: chọn encoder.onnx; giữ ba file temporal cạnh nó.", wraplength=290).pack(anchor="w", pady=6)
        self.mode = tk.StringVar(value="YOLO")
        for mode in ("YOLO", "YOLO + ConvLSTM"):
            radio = ttk.Radiobutton(sidebar, text=mode, variable=self.mode, value=mode)
            radio.pack(anchor="w", pady=2)
            self.controls.append(radio)
        ttk.Label(sidebar, text="Confidence YOLO").pack(anchor="w", pady=(8, 2))
        self.conf = tk.StringVar(value="0.25")
        confidence = ttk.Spinbox(sidebar, from_=.05, to=.95, increment=.05, textvariable=self.conf, width=10)
        confidence.pack(anchor="w")
        self.controls.append(confidence)
        buttons = ttk.Frame(sidebar)
        buttons.pack(fill="x", pady=14)
        self.start_button = ttk.Button(buttons, text="▶ Chạy", command=self.start)
        self.start_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.stop_button = ttk.Button(buttons, text="■ Dừng", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", fill="x", expand=True)
        ttk.Label(sidebar, text="Đỏ: vượt ngưỡng • Xanh: dưới ngưỡng\nY: confidence YOLO\nP: điểm drowning của ConvLSTM\nT: số mẫu của lần dự đoán / 32\nT < 32: dự đoán sớm", wraplength=290).pack(anchor="w")
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(right, bg="#101c2e", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self.redraw)
        self.position = ttk.Label(right, text="Chọn video và model rồi bấm Chạy.")
        self.position.pack(anchor="w", pady=8)
        self.progress = ttk.Progressbar(right, maximum=100)
        self.progress.pack(fill="x")
        metrics = ttk.Frame(outer)
        metrics.pack(fill="x", pady=14)
        self.metrics = []
        for i, title in enumerate(("FPS XỬ LÝ", "FPS HIỂN THỊ", "TRỄ KẾT QUẢ", "MẪU 10 FPS BỊ BỎ")):
            metrics.columnconfigure(i, weight=1)
            cell = ttk.Frame(metrics)
            cell.grid(row=0, column=i, sticky="w")
            ttk.Label(cell, text=title).pack(anchor="w")
            value = ttk.Label(cell, text="—", font=("Segoe UI", 18, "bold"))
            value.pack(anchor="w")
            self.metrics.append(value)
        self.backend = ttk.Label(outer, text="Thiết bị sẽ được chọn theo model: CUDA / DirectML / CPU.", wraplength=1150)
        self.backend.pack(anchor="w")
        self.status = ttk.Label(outer, text="Sẵn sàng", wraplength=1150)
        self.status.pack(anchor="w", pady=(4, 0))
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(10, self.poll)

    def file_field(self, parent, title, variable, folder, types):
        ttk.Label(parent, text=title).pack(anchor="w", pady=(8, 3))
        row = ttk.Frame(parent)
        row.pack(fill="x")
        entry = ttk.Entry(row, textvariable=variable)
        entry.pack(side="left", fill="x", expand=True)

        def browse():
            path = filedialog.askopenfilename(title=title, initialdir=ROOT / folder, filetypes=types)
            if path:
                variable.set(path)

        button = ttk.Button(row, text="Chọn…", command=browse, width=6)
        button.pack(side="left", padx=(5, 0))
        self.controls.extend((entry, button))

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        try:
            required = (self.video, self.yolo, self.convlstm) if self.mode.get() == "YOLO + ConvLSTM" else (self.video, self.yolo)
            for variable in required:
                if not variable.get().strip() or not Path(variable.get().strip()).expanduser().is_file():
                    raise ValueError("Chọn file video và model có trên máy.")
            confidence = float(self.conf.get())
            if not 0 <= confidence <= 1:
                raise ValueError("Confidence phải nằm trong [0, 1].")
        except ValueError as error:
            messagebox.showerror("Không thể chạy", str(error))
            return
        options = dict(video=Path(self.video.get().strip()).expanduser(), yolo=Path(self.yolo.get().strip()).expanduser(),
                       convlstm=Path(self.convlstm.get().strip()).expanduser(), mode=self.mode.get(), conf=confidence)
        for q in (self.frames, self.events):
            while not q.empty():
                q.get_nowait()
        self.stop_event = threading.Event()
        self.finished = False
        self.display_times.clear()
        self.photo = None
        self.canvas.delete("all")
        self.progress["value"] = 0
        for value in self.metrics:
            value.configure(text="—")
        self.backend.configure(text="Đang chọn thiết bị…")
        for widget in self.controls:
            widget.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.thread = threading.Thread(target=predict_video, args=(options, self.frames, self.events, self.stop_event), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.status.configure(text="Đang dừng sau frame hiện tại…")

    def redraw(self, event=None):
        if self.photo:
            self.canvas.delete("all")
            self.canvas.create_image(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2, image=self.photo)

    def poll(self):
        try:
            packet = self.frames.get_nowait()
        except queue.Empty:
            packet = None
        if packet:
            image = Image.fromarray(packet["image"])
            image.thumbnail((max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())), Image.Resampling.BILINEAR)
            self.photo = ImageTk.PhotoImage(image)
            self.redraw()
            now = time.perf_counter()
            self.display_times.append(now)
            fps = ((len(self.display_times) - 1) / (now - self.display_times[0]) if len(self.display_times) > 1 else 0)
            latency = max(0, now - packet["started"] - packet["timestamp"]) * 1000
            for widget, text in zip(self.metrics, (f"{packet['fps']:.1f}", f"{fps:.1f}", f"{latency:.0f} ms", str(packet["skipped"]))):
                widget.configure(text=text)
            self.position.configure(text=(f"{packet['timestamp']:.1f} / {packet['duration']:.1f}s • nguồn {packet['source_fps']:.2f} FPS → {packet['target_fps']:g} FPS • {packet['people']} box\n"
                                          f"YOLO {packet['detector_ms']:.0f} ms • tracking/crop/ConvLSTM/vẽ {packet['temporal_ms']:.0f} ms"))
            self.progress["value"] = min(100, packet["timestamp"] / packet["duration"] * 100)
        while not self.events.empty():
            kind, text = self.events.get_nowait()
            if kind == "done":
                self.finished = True
                self.stop_button.configure(state="disabled")
                if self.stop_event.is_set():
                    self.status.configure(text="Đã dừng.")
            elif kind == "backend":
                self.backend.configure(text=text)
            else:
                self.status.configure(text=f"Lỗi: {text}" if kind == "error" else text)
        # DirectML can take time releasing sessions after the last frame; do not offer a dead Start button.
        if self.finished and not self.thread.is_alive():
            self.finished = False
            for widget in self.controls:
                widget.configure(state="normal")
            self.start_button.configure(state="normal")
        self.root.after(10, self.poll)

    def close(self):
        self.stop_event.set()
        self.root.destroy()


def main():
    root = tk.Tk()
    DemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

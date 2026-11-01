"""ONNX adapters for the existing YOLO/tracking/cached-ConvLSTM demo."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results
from ultralytics.utils.nms import non_max_suppression


def metadata(path):
    import onnx
    graph = onnx.load(str(path), load_external_data=False)
    return {item.key: item.value for item in graph.metadata_props}


def session(path, device):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.log_severity_level = 3
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = False
    provider = ("DmlExecutionProvider" if device == "dml" else
                "CUDAExecutionProvider" if device.startswith("cuda") else "CPUExecutionProvider")
    if provider not in ort.get_available_providers():
        raise RuntimeError(f"Chưa có {provider}. AMD cần onnxruntime-directml; hoặc chọn CPU.")
    result = ort.InferenceSession(str(path), sess_options=options, providers=[provider])
    if result.get_providers()[0] != provider:
        raise RuntimeError(f"Không kích hoạt được {provider}")
    return result


def inspect(path):
    info = metadata(path)
    if info.get("kind") == '"convlstm"':
        if json.loads(info.get("part", '""')) != "encoder":
            raise ValueError("Chọn encoder.onnx; các file temporal phải ở cùng thư mục.")
        return dict(kind="convlstm", summary="ConvLSTM ONNX • encoder + 8/16/32 frame")
    if info.get("task") != "detect" or "names" not in info:
        raise ValueError("Cần YOLO detect ONNX hoặc encoder ONNX của dự án.")
    return dict(kind="yolo", summary=f"YOLO ONNX • {len(ast.literal_eval(info['names']))} lớp")


class Detector:
    task = "detect"

    def __init__(self, path, device):
        info = metadata(path)
        inspect(path)
        self.names = ast.literal_eval(info["names"])
        self.runtime = session(path, device)
        shape = self.runtime.get_inputs()[0].shape
        if not all(isinstance(x, int) for x in shape[2:]):
            raise ValueError("Export YOLO với dynamic=False để dùng app này.")
        self.height, self.width = shape[2:]

    def predict(self, frame, conf=.25, iou=.5, classes=None, agnostic_nms=False, **unused):
        height, width = frame.shape[:2]
        ratio = min(self.height / height, self.width / width)
        new_width, new_height = round(width * ratio), round(height * ratio)
        dw, dh = (self.width - new_width) / 2, (self.height - new_height) / 2
        left, top = round(dw - .1), round(dh - .1)
        resized = cv2.resize(frame, (new_width, new_height))
        image = cv2.copyMakeBorder(resized, top, round(dh + .1), left, round(dw + .1),
                                   cv2.BORDER_CONSTANT, value=(114, 114, 114))
        image = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
        output = self.runtime.run(None, {self.runtime.get_inputs()[0].name: image})[0]
        end2end = output.shape[-1] == 6
        boxes = non_max_suppression(torch.from_numpy(output.copy()), conf_thres=conf, iou_thres=iou,
                                    classes=classes, agnostic=agnostic_nms, nc=len(self.names), end2end=end2end)[0]
        if len(boxes):
            boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / ratio).clamp(0, width)
            boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / ratio).clamp(0, height)
        return [Results(frame, path="", names=self.names, boxes=boxes)]


def load_classifier(path, device, threshold=None):
    try:
        from .demo_engine import crop_person, normalize
    except ImportError:
        from demo_engine import crop_person, normalize
    path = Path(path)
    info = {key: json.loads(value) for key, value in metadata(path).items()}
    if info.get("part") != "encoder" or info.get("format_version") != 1:
        raise ValueError("Chọn encoder.onnx từ bộ export ConvLSTM.")
    if info["classes"] != ["drowning", "not_drowning"] or info["fps"] != 10:
        raise ValueError("ONNX class/FPS contract không tương thích.")
    runtimes = {}
    for steps in (8, 16, 32):
        head = path.parent / f"temporal-{steps}.onnx"
        meta = {key: json.loads(value) for key, value in metadata(head).items()}
        for key in ("checkpoint_sha256", "classes", "threshold", "fps", "image_size", "sequence_length", "crop_scale", "hidden_channels"):
            if info[key] != meta[key]:
                raise ValueError(f"Các file ONNX không cùng checkpoint: {head.name}")
        if meta.get("steps") != steps or meta.get("part") != "temporal":
            raise ValueError(f"Sai head thời gian: {head.name}")
        runtimes[steps] = session(head, device)
    encoder = session(path, device)

    class Model:
        temporal = SimpleNamespace(project_input=lambda x: x)

        def encoder(self, x):
            array = np.ascontiguousarray(x.cpu().numpy())
            return torch.from_numpy(encoder.run(None, {"input": array})[0])

        def classify_features(self, x, projected=False):
            if not projected:
                raise ValueError("ONNX temporal input must contain cached projected features")
            return torch.from_numpy(runtimes[x.shape[1]].run(None, {"input": np.ascontiguousarray(x.cpu().numpy())})[0])

    limit = float(info["threshold"] if threshold is None else threshold)
    if not 0 <= limit <= 1:
        raise ValueError("Ngưỡng ConvLSTM phải nằm trong [0, 1].")
    return SimpleNamespace(model=Model(), index=0, threshold=limit, length=info["sequence_length"],
                           size=info["image_size"], scale=info["crop_scale"], crop=crop_person,
                           normalize=normalize, device=torch.device("cpu"))

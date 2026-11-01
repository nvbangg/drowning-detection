"""Inference helpers shared by the desktop demo; no GUI or file output."""

import pickle
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torchvision.models import resnet18
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

FPS = 10


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, hidden_channels * 4, 3, padding=1)

    def forward(self, x, h, c):
        return self.step(self.gates(torch.cat([x, h], dim=1)), c)

    @staticmethod
    def step(gates, c):
        i, f, g, o = gates.chunk(4, dim=1)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        return o.sigmoid() * c.tanh(), c

    def project_input(self, x):
        return F.conv2d(x, self.gates.weight[:, :-self.hidden_channels], self.gates.bias,
                        padding=self.gates.padding)

    def forward_projected(self, projected, h, c):
        recurrent = F.conv2d(h, self.gates.weight[:, -self.hidden_channels:], padding=self.gates.padding)
        return self.step(projected + recurrent, c)

class PersonConvLSTM(nn.Module):
    def __init__(self, hidden_channels: int = 128, encoder_weights: str | Path | None = None):
        super().__init__()
        backbone = resnet18(weights=None)
        if encoder_weights:
            backbone.load_state_dict(torch.load(encoder_weights, map_location="cpu", weights_only=True))
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.temporal = ConvLSTMCell(512, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 2)

    def forward(self, clips):
        batch, length, channels, height, width = clips.shape
        features = self.encoder(clips.reshape(batch * length, channels, height, width))
        return self.classify_features(features.reshape(batch, length, *features.shape[1:]))

    def classify_features(self, features, projected=False):
        batch, _, _, height, width = features.shape
        h = features.new_zeros(batch, self.temporal.hidden_channels, height, width)
        c = torch.zeros_like(h)
        for frame in features.unbind(dim=1):
            h, c = (self.temporal.forward_projected(frame, h, c) if projected
                    else self.temporal(frame, h, c))
        return self.classifier(h.mean(dim=(2, 3)))

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def normalize(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    array = (array - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1))

def crop_person(image: Image.Image, box, scale: float = 1.5, size: int = 224):
    x1, y1, x2, y2 = box
    raw_side = max(x2 - x1, y2 - y1) * scale
    left, top = round((x1 + x2 - raw_side) / 2), round((y1 + y2 - raw_side) / 2)
    side = max(1, round(raw_side))
    # Pillow pads an out-of-frame crop; use the same fill as YOLO letterboxing.
    canvas = Image.new("RGB", (side, side), (114, 114, 114))
    area = max(0, left), max(0, top), min(image.width, left + side), min(image.height, top + side)
    if area[2] <= area[0] or area[3] <= area[1]:
        raise ValueError(f"Box does not intersect the image: {box}")
    canvas.paste(image.crop(area), (area[0] - left, area[1] - top))
    return canvas.resize((size, size), Image.Resampling.BILINEAR)


def label_name(value):
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def safe_checkpoint(path):
    """Read state-dict checkpoints; do not allowlist arbitrary pickled model classes."""
    core = getattr(np, "_core", None)
    if core is None:
        core = np.core
    allowed = [np.dtype, core.multiarray.scalar, type(np.dtype("float64"))]
    try:
        with torch.serialization.safe_globals(allowed):
            return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        return None  # Ultralytics checkpoints use their own official model loader.


def inspect_checkpoint(path):
    """Infer supported architecture from contents, never from the filename."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {path}")
    if path.suffix.lower() == ".onnx":
        try:
            from .onnx_models import inspect
        except ImportError:
            from onnx_models import inspect
        return inspect(path)
    checkpoint = safe_checkpoint(path)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model", checkpoint.get("state_dict"))
        if isinstance(state, dict) and any(key.startswith("temporal.gates.") for key in state):
            required = {"classes", "sequence_length", "hidden_channels"}
            if not required <= checkpoint.keys():
                raise ValueError("Checkpoint ConvLSTM thiếu metadata: classes, sequence_length hoặc hidden_channels.")
            return {"kind": "convlstm", "summary":
                    f"ConvLSTM • {checkpoint['sequence_length']} frame @ {checkpoint.get('fps', FPS):g} FPS"}
    from ultralytics import YOLO

    try:
        model = YOLO(str(path))
    except Exception as error:
        raise ValueError("Không nhận ra checkpoint YOLO hoặc ResNet18 + ConvLSTM tương thích.") from error
    return {"kind": "yolo", "summary": f"YOLO • {model.task} • {len(model.names)} lớp"}


def action_classes(names):
    """Require both water-action classes, regardless of their numeric class ID order."""
    aliases = {"drown": "drowning", "swim": "not_drowning", "swimming": "not_drowning"}
    classes = {int(k): aliases.get(label_name(v), label_name(v)) for k, v in names.items()}
    if len(classes) != 2 or set(classes.values()) != {"drowning", "not_drowning"}:
        raise ValueError("Cần YOLO hai lớp drowning/not_drowning; không dùng model person.")
    return classes


def load_classifier(path, device, threshold=None):
    if Path(path).suffix.lower() == ".onnx":
        try:
            from .onnx_models import load_classifier as load_onnx
        except ImportError:
            from onnx_models import load_classifier as load_onnx
        return load_onnx(path, str(device), threshold)
    checkpoint = safe_checkpoint(path)
    if not isinstance(checkpoint, dict):
        raise ValueError("Cần checkpoint ConvLSTM dạng state_dict kèm metadata.")
    classes = [label_name(c) for c in checkpoint["classes"]]
    drowning = [i for i, name in enumerate(classes) if name in {"drowning", "drown"}]
    normal = [name for name in classes if name in {"not_drowning", "swim", "swimming"}]
    if len(classes) != 2 or len(drowning) != 1 or len(normal) != 1:
        raise ValueError(f"Unsupported ConvLSTM classes: {classes}")
    if float(checkpoint.get("fps", FPS)) != FPS:
        raise ValueError("Classifier must have been trained at 10 FPS")
    model = PersonConvLSTM(int(checkpoint["hidden_channels"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    model.encoder.to(memory_format=torch.channels_last)
    limit = float(checkpoint.get("threshold", 0.5) if threshold is None else threshold)
    if not 0 <= limit <= 1:
        raise ValueError("Drowning threshold must be between 0 and 1")
    return SimpleNamespace(model=model, index=drowning[0], threshold=limit,
                           length=int(checkpoint["sequence_length"]),
                           size=int(checkpoint.get("image_size", 224)),
                           scale=float(checkpoint.get("crop_scale", 1.5)),
                           crop=crop_person, normalize=normalize, device=device)


def draw_box(frame, box, text, color):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(round, box)
    x1, x2 = max(0, x1), min(width - 1, x2)
    y1, y2 = max(0, y1), min(height - 1, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    tx, ty = min(x1, max(0, width - tw - 6)), max(th + 5, y1 - 5)
    cv2.rectangle(frame, (tx, ty - th - 4), (min(width - 1, tx + tw + 5), ty + baseline), (20, 25, 35), -1)
    cv2.putText(frame, text, (tx + 2, ty), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


class TemporalPeople:
    def __init__(self, classifier=None):
        self.classifier = classifier
        self.people = {}
        self.predictions = 0
        self.tracker = BYTETracker(SimpleNamespace(track_high_thresh=0.25, track_low_thresh=0.1,
                                  new_track_thresh=0.25, track_buffer=30,
                                  match_thresh=0.8, fuse_score=True))

    def update(self, frame, result, timestamp):
        c = self.classifier
        # Keep only recent observations; an interrupted temporal sequence must restart.
        self.people = {key: person for key, person in self.people.items() if timestamp - person["seen"] <= 2.1 / FPS}
        # Track all supported action classes as people; a YOLO class flip must not reset an ID.
        boxes = result.boxes.data.detach().cpu().numpy().copy()
        boxes[:, -1] = 0
        tracks = self.tracker.update(Boxes(boxes, result.orig_shape), img=frame)
        current = []
        for track in tracks:
            person_id = int(track[4])
            # Use the actual current detection for cropping, not a predicted lost-track box.
            detection = int(track[-1])
            box = result.boxes.xyxy[detection].cpu().tolist()
            person = self.people.setdefault(person_id, {
                "features": deque(maxlen=c.length if c else 0), "seen": timestamp,
                "probability": None, "since_prediction": 0,
                "prediction_length": 0,
            })
            gap = timestamp - person["seen"]
            if gap > 1.5 / FPS and person["features"]:
                # Bridge one missed observation with the preceding feature, matching training augmentation.
                person["features"].append(person["features"][-1])
                person["since_prediction"] += 1
            yolo_label = label_name(result.names[int(result.boxes.cls[detection])])
            yolo_label = {"drown": "drowning", "swim": "not_drowning",
                          "swimming": "not_drowning"}.get(yolo_label, yolo_label)
            person.update(seen=timestamp, box=box, yolo_label=yolo_label,
                          yolo_score=float(result.boxes.conf[detection]))
            current.append((person_id, person))
        if current and c is not None:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            crops = torch.stack([c.normalize(c.crop(image, p["box"], c.scale, c.size))
                                 for _, p in current]).to(c.device, memory_format=torch.channels_last)
            with torch.inference_mode(), torch.autocast(c.device.type, enabled=c.device.type == "cuda"):
                # Cache W_x*x + bias once. Each sliding window then recomputes only W_h*h.
                # Small batches bound ResNet's working memory in crowded full-frame videos.
                features = torch.cat([c.model.temporal.project_input(c.model.encoder(chunk))
                                      for chunk in crops.split(8)])
                ready = {}
                for (_, person), feature in zip(current, features):
                    person["features"].append(feature)
                    person["since_prediction"] += 1
                    length = len(person["features"])
                    # Prefix results are provisional; group equal lengths instead of padding with future frames.
                    stage = max((n for n in {min(8, c.length), min(16, c.length), c.length}
                                 if n <= length), default=0)
                    if stage and (person["prediction_length"] < stage or (
                            stage == c.length and person["since_prediction"] >= 8)):
                        ready.setdefault(stage, []).append(person)
                for length, people in ready.items():
                    windows = torch.stack([torch.stack(list(p["features"])[:length]) for p in people])
                    probabilities = c.model.classify_features(windows, projected=True).float().softmax(1)[:, c.index].tolist()
                    for person, probability in zip(people, probabilities):
                        person.update(probability=probability, since_prediction=0, prediction_length=length)
                    self.predictions += len(people)
        output = frame.copy()
        for person_id, person in current:
            if person["probability"] is None:
                # Both YOLO classes locate people and supply the initial state, until ConvLSTM is ready.
                state = person["yolo_label"]
                text = f"ID {person_id} Y={person['yolo_score']:.3f}"
                if c:
                    text += f" T={len(person['features'])}/{min(8, c.length)}"
                color = ((0, 0, 255) if state == "drowning" else
                         (30, 210, 60) if state == "not_drowning" else (180, 150, 90))
                draw_box(output, person["box"], text, color)
            else:
                # Once available, the temporal state replaces YOLO's state; YOLO keeps supplying the boxes.
                risk = person["probability"] >= c.threshold
                text = (f"ID {person_id} Y={person['yolo_score']:.3f} P={person['probability']:.4f} "
                        f"T={person['prediction_length']}/{c.length}")
                color = (0, 0, 255) if risk else (30, 210, 60)
                draw_box(output, person["box"], text, color)
        return output

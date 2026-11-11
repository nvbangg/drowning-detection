# Copyright (c) 2026 nvbangg (github.com/nvbangg)

import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import signal
import sys
import time
import tkinter as tk
from tkinter import filedialog, simpledialog

import cv2
import numpy as np

# ==============================================================================
# 1. PATHS & DATASET CONFIG
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
PROGRESS_FILE = ROOT_DIR / "review_progress.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ==============================================================================
# 2. WINDOW & UI CONFIG
# ==============================================================================
DEFAULT_WINDOW_SIZE = (1250, 780)
MIN_CANVAS_SIZE = (700, 450)
MIN_RESIZE_SIZE = (300, 250)
LOOP_DELAY_MS = 15

# Panel layout
PANEL_HEIGHT_RATIO = 0.095
PANEL_HEIGHT_RANGE = (70, 105)

# UI scaling
UI_SCALE_REF_WIDTH = 1250.0
UI_SCALE_RANGE = (1.0, 1.6)
BOX_FONT_SCALE_FACTOR = 0.55
BUTTON_FONT_SCALE_FACTOR = 0.58
BOX_THICKNESS_FACTOR = 2.2
SELECTED_EXTRA_THICKNESS = 2

# Interaction thresholds
MIN_DRAW_BOX_SIZE = 6
DRAG_THRESHOLD = 5

# Keyboard key codes (OpenCV platform-specific extended codes)
KEY_ESC = 27
KEY_SPACE = 32
KEYS_ARROW_LEFT = (2424832, 81, 65361)
KEYS_ARROW_RIGHT = (2555904, 83, 65363)
KEYS_DELETE = (3014656, 127, 8, 65535)

# Continuous key navigation config
HOLD_INIT_DELAY_SEC = 0.25
HOLD_REPEAT_INTERVAL_SEC = 0.35
AUTO_PLAY_INTERVAL_SEC = HOLD_REPEAT_INTERVAL_SEC
VK_SPACE = 0x20
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
GA_ROOT = 2
HAS_WIN32 = hasattr(ctypes, "windll")

# Class definitions (0: Drowning, 1: Not Drowning)
CLASS_NAMES = {0: "Drowning", 1: "Not Drowning"}
CLASS_COLORS = {
    0: (60, 76, 231),
    1: (113, 204, 46),
}

# UI colors (BGR)
COLOR_SELECTED = (15, 220, 255)
COLOR_DRAWING = (255, 230, 0)
COLOR_PANEL_BG = (28, 28, 30)
COLOR_BUTTON_BG = (50, 50, 54)
COLOR_TEXT = (240, 240, 240)
COLOR_BORDER = (100, 100, 105)
COLOR_BORDER_DIM = (90, 90, 95)
COLOR_SEPARATOR = (80, 80, 85)
COLOR_PANEL_LINE = (70, 70, 75)
COLOR_INDEX_BG = (18, 18, 20)
COLOR_INDEX_BORDER = (0, 180, 240)
COLOR_INDEX_TEXT = (0, 220, 255)
COLOR_BADGE_BG = (20, 20, 20)
COLOR_DELETE_BOX_BG = (35, 35, 115)
COLOR_DELETE_IMAGE_BG = (35, 35, 160)


def choose_dataset_directory() -> Path | None:
    """Select the dataset root containing images/ and labels/."""
    root = tk.Tk()
    try:
        root.withdraw()
        root.attributes("-topmost", True)
        initial = ROOT_DIR / "datasets"
        selected = filedialog.askdirectory(
            parent=root,
            title="Chọn thư mục dataset YOLO (chứa images và labels)",
            initialdir=str(initial if initial.is_dir() else ROOT_DIR),
            mustexist=True,
        )
        return Path(selected).resolve() if selected else None
    finally:
        root.destroy()


def load_class_names(dataset_dir: Path) -> dict[int, str]:
    """Read class IDs as stored, without remapping any annotation files."""
    yaml_path = next(
        (dataset_dir / name for name in ("data.yaml", "data.yml")
         if (dataset_dir / name).is_file()), None,
    )
    if yaml_path:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("Reading data.yaml requires: python -m pip install PyYAML") from exc
        try:
            config = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
            names = config.get("names") if isinstance(config, dict) else None
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"Cannot read {yaml_path}: {exc}") from exc
        if names is None:
            raise ValueError(f"Missing 'names' in {yaml_path}")
        if isinstance(names, list):
            names = dict(enumerate(names))
    elif (dataset_dir / "classes.txt").is_file():
        names = dict(enumerate(
            line.strip() for line in (dataset_dir / "classes.txt").read_text(
                encoding="utf-8-sig").splitlines() if line.strip()
        ))
    else:
        return dict(CLASS_NAMES)
    try:
        names = {int(i): name.strip() for i, name in names.items()}
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Class names must be a list or a mapping of IDs to names") from exc
    if set(names) != {0, 1} or any(not name for name in names.values()):
        raise ValueError("This reviewer requires exactly two named classes, IDs 0 and 1")
    return dict(sorted(names.items()))


class DatasetReviewer:
    """Manages YOLO dataset review with responsive UI, annotation editing, and navigation."""

    def __init__(self, dataset_dir: Path, dataset_name: str | None = None):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name or dataset_dir.name
        self.window_name = f"{self.dataset_name} Review"
        self.progress_file = PROGRESS_FILE
        self.class_names = load_class_names(dataset_dir)
        self.class_colors = dict(CLASS_COLORS)
        for class_id, name in self.class_names.items():
            normalized = name.lower().replace("_", " ").replace("-", " ")
            if normalized in ("drown", "drowning"):
                self.class_colors[class_id] = CLASS_COLORS[0]
            elif normalized in ("swim", "swimming", "not drowning", "not drown"):
                self.class_colors[class_id] = CLASS_COLORS[1]

        self.image_paths = self._collect_image_paths()
        self.current_index = 0
        self.annotations = []
        self.selected_index = None
        self.active_class_id = 0
        self.running = True
        self.show_labels = True

        self.is_dragging = False
        self.drag_start = None
        self.drag_end = None
        self.drag_moved = False

        self.window_width = 0
        self.window_height = 0
        self.display_width = 0
        self.display_height = 0
        self.offset_x = 0
        self.offset_y = 0
        self.panel_height = PANEL_HEIGHT_RANGE[0]

        self.original_image = None
        self.cached_canvas = None
        self.button_rects = []
        self.index_box_rect = None

        self.nav_key_down = None
        self.nav_press_start_time = 0.0
        self.prev_nav_repeat_time = 0.0
        self.is_nav_holding = False
        self.is_auto_playing = False
        self.prev_auto_play_time = 0.0

        self._load_progress()
        self._load_current_image()

    def _collect_image_paths(self) -> list[Path]:
        images_dir = self.dataset_dir / "images"
        if not images_dir.is_dir():
            return []
        return sorted(
            [
                p
                for p in images_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=lambda p: str(p).lower(),
        )

    # --- Progress persistence ---

    def _get_dataset_key(self) -> str:
        try:
            return str(
                self.dataset_dir.resolve().relative_to(ROOT_DIR.resolve())
            ).replace("\\", "/")
        except ValueError:
            return str(self.dataset_dir.resolve()).replace("\\", "/")

    def _load_progress(self):
        if not self.image_paths:
            return

        dataset_info = None
        if self.progress_file.exists():
            try:
                data = json.loads(self.progress_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    key = self._get_dataset_key()
                    dataset_info = data.get(key)
                    if dataset_info is None:
                        dataset_info = data.get(self.dataset_name) or data.get(
                            str(self.dataset_dir.resolve())
                        )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

        legacy_file = self.dataset_dir / "review_progress.json"
        if dataset_info is None and legacy_file.exists():
            try:
                dataset_info = json.loads(legacy_file.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

        if dataset_info is None:
            return

        try:
            if isinstance(dataset_info, dict):
                index = int(dataset_info.get("index", 0))
            elif isinstance(dataset_info, (int, float)):
                index = int(dataset_info)
            else:
                return
            self.current_index = max(0, min(index, len(self.image_paths) - 1))
        except (ValueError, TypeError):
            pass

    def sync_progress(self):
        try:
            data = {}
            if self.progress_file.exists():
                try:
                    loaded = json.loads(self.progress_file.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        data = loaded
                except (OSError, ValueError, json.JSONDecodeError):
                    data = {}

            key = self._get_dataset_key()
            data[key] = {"index": self.current_index}
            self.progress_file.write_text(
                json.dumps(data, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    # --- Label file I/O ---

    def _get_label_path(self, image_path: Path) -> Path | None:
        try:
            rel = image_path.relative_to(self.dataset_dir / "images")
            return (self.dataset_dir / "labels" / rel).with_suffix(".txt")
        except ValueError:
            return None

    def _read_annotations(self, label_path: Path | None) -> list[dict]:
        if not label_path or not label_path.exists():
            return []
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

        annotations = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                continue
            if class_id in self.class_names:
                annotations.append(
                    {
                        "class_id": class_id,
                        "x": max(0.0, min(1.0, x)),
                        "y": max(0.0, min(1.0, y)),
                        "w": max(0.0, min(1.0, w)),
                        "h": max(0.0, min(1.0, h)),
                    }
                )
        return annotations

    def _write_annotations(self):
        if not self.image_paths:
            return
        label_path = self._get_label_path(self.image_paths[self.current_index])
        if not label_path:
            return
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{ann['class_id']} {ann['x']:.6f} {ann['y']:.6f} {ann['w']:.6f} {ann['h']:.6f}"
            for ann in self.annotations
        ]
        try:
            label_path.write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
        except OSError:
            pass

    # --- Image loading and navigation ---

    def _update_window_title(self):
        if self.image_paths:
            try:
                title = (
                    f"{self.window_name} - {self.image_paths[self.current_index].name}"
                )
                cv2.setWindowTitle(self.window_name, title)
            except Exception:
                pass

    def _load_current_image(self):
        if not self.image_paths:
            self.original_image = None
            self.annotations = []
            return
        image_path = self.image_paths[self.current_index]
        self.original_image = cv2.imread(str(image_path))
        self.selected_index = None
        self.annotations = self._read_annotations(self._get_label_path(image_path))
        self.cached_canvas = None
        self._update_window_title()

    def prev_image(self, sync_disk: bool = True):
        if self.image_paths and self.current_index > 0:
            self.current_index -= 1
            if sync_disk:
                self.sync_progress()
            self._load_current_image()

    def next_image(self, sync_disk: bool = True):
        if self.image_paths and self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            if sync_disk:
                self.sync_progress()
            self._load_current_image()

    def _jump_to_index_dialog(self):
        if not self.image_paths:
            return
        self.stop_auto_play()
        try:
            temp_root = tk.Tk()
            temp_root.withdraw()
            temp_root.attributes("-topmost", True)
            target = simpledialog.askinteger(
                "Jump to Image",
                f"Enter image index (1 - {len(self.image_paths)}):",
                initialvalue=self.current_index + 1,
                minvalue=1,
                maxvalue=len(self.image_paths),
                parent=temp_root,
            )
            temp_root.destroy()
            if target is not None:
                self.current_index = target - 1
                self.sync_progress()
                self._load_current_image()
        except Exception:
            pass

    # --- Annotation manipulation ---

    def _annotation_to_rect(self, annotation: dict) -> tuple[int, int, int, int]:
        """Convert YOLO normalized annotation to clamped display pixel rectangle."""
        half_w, half_h = annotation["w"] / 2.0, annotation["h"] / 2.0
        left = self.offset_x + int(
            round((annotation["x"] - half_w) * self.display_width)
        )
        top = self.offset_y + int(
            round((annotation["y"] - half_h) * self.display_height)
        )
        right = self.offset_x + int(
            round((annotation["x"] + half_w) * self.display_width)
        )
        bottom = self.offset_y + int(
            round((annotation["y"] + half_h) * self.display_height)
        )
        max_right = self.offset_x + self.display_width - 1
        max_bottom = self.offset_y + self.display_height - 1
        return (
            max(self.offset_x, min(left, max_right)),
            max(self.offset_y, min(top, max_bottom)),
            max(self.offset_x, min(right, max_right)),
            max(self.offset_y, min(bottom, max_bottom)),
        )

    def _add_box_from_drag(self, start: tuple[int, int], end: tuple[int, int]):
        if self.display_width <= 0 or self.display_height <= 0:
            return
        left = max(self.offset_x, min(start[0], end[0]))
        right = min(self.offset_x + self.display_width - 1, max(start[0], end[0]))
        top = max(self.offset_y, min(start[1], end[1]))
        bottom = min(self.offset_y + self.display_height - 1, max(start[1], end[1]))

        if right - left < MIN_DRAW_BOX_SIZE or bottom - top < MIN_DRAW_BOX_SIZE:
            return

        center_x = ((left + right) / 2.0 - self.offset_x) / self.display_width
        center_y = ((top + bottom) / 2.0 - self.offset_y) / self.display_height

        self.annotations.append(
            {
                "class_id": self.active_class_id,
                "x": max(0.0, min(1.0, center_x)),
                "y": max(0.0, min(1.0, center_y)),
                "w": max(0.0, min(1.0, (right - left) / float(self.display_width))),
                "h": max(0.0, min(1.0, (bottom - top) / float(self.display_height))),
            }
        )
        self.selected_index = len(self.annotations) - 1
        self._write_annotations()
        self.cached_canvas = None

    def set_selected_class(self, new_class_id: int):
        self.active_class_id = new_class_id
        if self.selected_index is not None and 0 <= self.selected_index < len(
            self.annotations
        ):
            self.annotations[self.selected_index]["class_id"] = new_class_id
            self._write_annotations()
        elif self.annotations:
            for annotation in self.annotations:
                annotation["class_id"] = new_class_id
            self._write_annotations()
        self.cached_canvas = None

    def delete_selected_box(self):
        if self.selected_index is None or not (
            0 <= self.selected_index < len(self.annotations)
        ):
            return
        self.annotations.pop(self.selected_index)
        self.selected_index = (
            min(self.selected_index, len(self.annotations) - 1)
            if self.annotations
            else None
        )
        self._write_annotations()
        self.cached_canvas = None

    def delete_current_image(self):
        if not self.image_paths:
            return
        image_path = self.image_paths[self.current_index]
        label_path = self._get_label_path(image_path)
        try:
            image_path.unlink(missing_ok=True)
            if label_path:
                label_path.unlink(missing_ok=True)
        except OSError as err:
            print(f"Error deleting file: {err}")
            return
        self.image_paths.pop(self.current_index)
        self.current_index = (
            min(self.current_index, len(self.image_paths) - 1)
            if self.image_paths
            else 0
        )
        self.sync_progress()
        self._load_current_image()

    # --- Window sizing ---

    @staticmethod
    def get_init_window_size() -> tuple[int, int]:
        try:
            temp_root = tk.Tk()
            temp_root.withdraw()
            screen_width = temp_root.winfo_screenwidth()
            screen_height = temp_root.winfo_screenheight()
            temp_root.destroy()
            return (
                max(850, min(1400, int(screen_width * 0.85))),
                max(520, min(860, int(screen_height * 0.82))),
            )
        except Exception:
            return DEFAULT_WINDOW_SIZE

    def update_window_size(self, width: int, height: int):
        if width < MIN_RESIZE_SIZE[0] or height < MIN_RESIZE_SIZE[1]:
            return
        if width != self.window_width or height != self.window_height:
            self.window_width = width
            self.window_height = height
            self.cached_canvas = None

    # --- Canvas rendering ---

    def _build_canvas(self):
        if self.original_image is None:
            return None

        original_height, original_width = self.original_image.shape[:2]
        canvas_width = max(
            MIN_CANVAS_SIZE[0], self.window_width or DEFAULT_WINDOW_SIZE[0]
        )
        canvas_height = max(
            MIN_CANVAS_SIZE[1], self.window_height or DEFAULT_WINDOW_SIZE[1]
        )

        self.panel_height = max(
            PANEL_HEIGHT_RANGE[0],
            min(PANEL_HEIGHT_RANGE[1], int(round(canvas_height * PANEL_HEIGHT_RATIO))),
        )
        available_height = canvas_height - self.panel_height
        scale = min(
            canvas_width / float(original_width),
            available_height / float(original_height),
        )
        self.display_width = max(10, int(round(original_width * scale)))
        self.display_height = max(10, int(round(original_height * scale)))
        self.offset_x = (canvas_width - self.display_width) // 2
        self.offset_y = (available_height - self.display_height) // 2

        canvas = np.full(
            (canvas_height, canvas_width, 3), COLOR_PANEL_BG, dtype=np.uint8
        )
        scaled_image = cv2.resize(
            self.original_image,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA,
        )
        canvas[
            self.offset_y : self.offset_y + self.display_height,
            self.offset_x : self.offset_x + self.display_width,
        ] = scaled_image

        ui_scale = max(
            UI_SCALE_RANGE[0], min(UI_SCALE_RANGE[1], canvas_width / UI_SCALE_REF_WIDTH)
        )
        box_font_scale = BOX_FONT_SCALE_FACTOR * ui_scale
        box_thickness = max(2, int(round(BOX_THICKNESS_FACTOR * ui_scale)))

        self._draw_bounding_boxes(canvas, box_font_scale, box_thickness)
        self._draw_toolbar(canvas, canvas_width, canvas_height, ui_scale)

        self.cached_canvas = canvas
        return canvas

    def _draw_bounding_boxes(self, canvas, font_scale: float, base_thickness: int):
        for index, annotation in enumerate(self.annotations):
            box_left, box_top, box_right, box_bottom = self._annotation_to_rect(
                annotation
            )
            is_selected = index == self.selected_index
            color = (
                COLOR_SELECTED
                if is_selected
                else self.class_colors.get(annotation["class_id"], (0, 255, 0))
            )
            thickness = (
                base_thickness + SELECTED_EXTRA_THICKNESS
                if is_selected
                else base_thickness
            )
            cv2.rectangle(
                canvas, (box_left, box_top), (box_right, box_bottom), color, thickness
            )

            if not self.show_labels:
                continue

            badge_text = self.class_names[annotation["class_id"]]
            (text_width, text_height), baseline = cv2.getTextSize(
                badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )

            badge_bottom = box_top - 4
            badge_top = badge_bottom - text_height - baseline - 4
            if badge_top < self.offset_y:
                badge_top = box_top + 4
                badge_bottom = badge_top + text_height + baseline + 4

            badge_left = max(
                self.offset_x,
                min(box_left, self.offset_x + self.display_width - text_width - 8),
            )
            badge_right = badge_left + text_width + 8

            cv2.rectangle(
                canvas,
                (badge_left, badge_top),
                (badge_right, badge_bottom),
                COLOR_BADGE_BG,
                -1,
            )
            cv2.rectangle(
                canvas, (badge_left, badge_top), (badge_right, badge_bottom), color, 1
            )
            cv2.putText(
                canvas,
                badge_text,
                (badge_left + 4, badge_bottom - baseline - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                COLOR_TEXT,
                1,
                cv2.LINE_AA,
            )

    def _draw_toolbar(
        self, canvas, canvas_width: int, canvas_height: int, ui_scale: float
    ):
        panel_top = canvas_height - self.panel_height
        cv2.line(canvas, (0, panel_top), (canvas_width, panel_top), COLOR_PANEL_LINE, 2)

        button_top = panel_top + int(round(10 * ui_scale))
        button_bottom = canvas_height - int(round(10 * ui_scale))
        button_height = button_bottom - button_top
        button_font_scale = BUTTON_FONT_SCALE_FACTOR * ui_scale
        cursor_x = int(round(15 * ui_scale))
        spacing = int(round(12 * ui_scale))
        self.button_rects = []

        def draw_button(
            text: str,
            action_id: str,
            background: tuple[int, int, int],
            border: tuple[int, int, int],
            border_thickness: int = 1,
            text_color: tuple[int, int, int] = COLOR_TEXT,
            padding: int = 26,
            min_width: int = 0,
        ):
            nonlocal cursor_x
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, button_font_scale, 1
            )
            button_width = max(min_width, text_width + int(round(padding * ui_scale)))
            left, right = cursor_x, cursor_x + button_width
            cv2.rectangle(
                canvas, (left, button_top), (right, button_bottom), background, -1
            )
            cv2.rectangle(
                canvas,
                (left, button_top),
                (right, button_bottom),
                border,
                border_thickness,
            )
            cv2.putText(
                canvas,
                text,
                (
                    left + (button_width - text_width) // 2,
                    button_top + (button_height + text_height) // 2,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                button_font_scale,
                text_color,
                1,
                cv2.LINE_AA,
            )
            self.button_rects.append(
                (action_id, left, button_top, right, button_bottom)
            )
            cursor_x += button_width + spacing

        def draw_separator():
            nonlocal cursor_x
            cv2.line(
                canvas,
                (cursor_x, button_top + 4),
                (cursor_x, button_bottom - 4),
                COLOR_SEPARATOR,
                1,
            )
            cursor_x += spacing + 4

        # Toggle label visibility button
        (hide_lbl_w, _), _ = cv2.getTextSize(
            "Hide Label", cv2.FONT_HERSHEY_SIMPLEX, button_font_scale, 1
        )
        (show_lbl_w, _), _ = cv2.getTextSize(
            "Show Label", cv2.FONT_HERSHEY_SIMPLEX, button_font_scale, 1
        )
        label_btn_min_width = max(hide_lbl_w, show_lbl_w) + int(round(14 * ui_scale))

        label_btn_text = "Hide Label" if self.show_labels else "Show Label"
        label_btn_bg = (38, 55, 38) if self.show_labels else COLOR_BUTTON_BG
        label_btn_border = (60, 160, 60) if self.show_labels else COLOR_BORDER
        draw_button(
            label_btn_text,
            "TOGGLE_LABELS",
            label_btn_bg,
            label_btn_border,
            padding=14,
            min_width=label_btn_min_width,
        )

        draw_separator()

        draw_button("< Prev", "PREV", COLOR_BUTTON_BG, COLOR_BORDER)

        # Index position box (clickable to jump)
        position_text = f" {self.current_index + 1} / {len(self.image_paths)} "
        (pos_text_width, pos_text_height), _ = cv2.getTextSize(
            position_text, cv2.FONT_HERSHEY_SIMPLEX, button_font_scale, 1
        )
        position_box_width = max(
            pos_text_width + int(round(22 * ui_scale)), int(round(110 * ui_scale))
        )
        position_left, position_right = cursor_x, cursor_x + position_box_width
        cv2.rectangle(
            canvas,
            (position_left, button_top),
            (position_right, button_bottom),
            COLOR_INDEX_BG,
            -1,
        )
        cv2.rectangle(
            canvas,
            (position_left, button_top),
            (position_right, button_bottom),
            COLOR_INDEX_BORDER,
            2,
        )
        cv2.putText(
            canvas,
            position_text,
            (
                position_left + (position_box_width - pos_text_width) // 2,
                button_top + (button_height + pos_text_height) // 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            button_font_scale,
            COLOR_INDEX_TEXT,
            1,
            cv2.LINE_AA,
        )
        self.index_box_rect = (position_left, button_top, position_right, button_bottom)
        cursor_x += position_box_width + spacing

        draw_button("Next >", "NEXT", COLOR_BUTTON_BG, COLOR_BORDER)

        (auto_text_w, _), _ = cv2.getTextSize(
            "Auto Play", cv2.FONT_HERSHEY_SIMPLEX, button_font_scale, 1
        )
        auto_btn_min_width = auto_text_w + int(round(18 * ui_scale))

        auto_text = "Pause" if self.is_auto_playing else "Auto Play"
        auto_bg = (35, 120, 35) if self.is_auto_playing else (40, 50, 40)
        auto_border = (80, 220, 80) if self.is_auto_playing else (60, 140, 60)
        draw_button(
            auto_text,
            "AUTO_PLAY",
            auto_bg,
            auto_border,
            border_thickness=2 if self.is_auto_playing else 1,
            padding=18,
            min_width=auto_btn_min_width,
        )

        draw_separator()

        active_class = self.active_class_id
        if self.selected_index is not None and 0 <= self.selected_index < len(
            self.annotations
        ):
            active_class = self.annotations[self.selected_index]["class_id"]

        for class_id, class_name in self.class_names.items():
            is_active = active_class == class_id
            draw_button(
                class_name,
                f"CLASS_{class_id}",
                self.class_colors[class_id],
                (255, 255, 255) if is_active else COLOR_BORDER_DIM,
                2 if is_active else 1,
                padding=30,
            )

        draw_separator()

        draw_button("Delete Box", "DEL_BOX", COLOR_DELETE_BOX_BG, COLOR_BORDER)
        draw_button("Delete Image", "DEL_IMG", COLOR_DELETE_IMAGE_BG, COLOR_BORDER)

        # Box count badge in bottom-right corner
        box_count_text = f"Boxes: {len(self.annotations)}"
        (count_w, count_h), _ = cv2.getTextSize(
            box_count_text, cv2.FONT_HERSHEY_SIMPLEX, button_font_scale, 1
        )
        count_box_w = count_w + int(round(22 * ui_scale))
        count_right = canvas_width - int(round(15 * ui_scale))
        count_left = count_right - count_box_w
        if count_left > cursor_x + spacing:
            cv2.rectangle(
                canvas,
                (count_left, button_top),
                (count_right, button_bottom),
                COLOR_INDEX_BG,
                -1,
            )
            cv2.rectangle(
                canvas,
                (count_left, button_top),
                (count_right, button_bottom),
                COLOR_BORDER_DIM,
                1,
            )
            cv2.putText(
                canvas,
                box_count_text,
                (
                    count_left + (count_box_w - count_w) // 2,
                    button_top + (button_height + count_h) // 2,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                button_font_scale,
                COLOR_TEXT,
                1,
                cv2.LINE_AA,
            )

    def get_display_canvas(self):
        if self.cached_canvas is None:
            self._build_canvas()
        if self.cached_canvas is None:
            return None

        if self.is_dragging and self.drag_moved and self.drag_start and self.drag_end:
            overlay = self.cached_canvas.copy()
            cv2.rectangle(overlay, self.drag_start, self.drag_end, COLOR_DRAWING, 2)
            if self.show_labels:
                label = self.class_names[self.active_class_id]
                label_x = min(self.drag_start[0], self.drag_end[0])
                label_y = max(20, min(self.drag_start[1], self.drag_end[1]) - 6)
                cv2.putText(
                    overlay,
                    label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COLOR_DRAWING,
                    1,
                    cv2.LINE_AA,
                )
            return overlay

        return self.cached_canvas

    # --- Event handling ---

    def on_mouse_event(self, event: int, x: int, y: int, flags: int, param):
        if self.original_image is None:
            return

        canvas_height = max(
            MIN_CANVAS_SIZE[1], self.window_height or DEFAULT_WINDOW_SIZE[1]
        )
        panel_top = canvas_height - self.panel_height

        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
            if y < panel_top:
                in_image_x = self.offset_x <= x <= self.offset_x + self.display_width
                in_image_y = self.offset_y <= y <= self.offset_y + self.display_height
                if in_image_x and in_image_y:
                    self.is_dragging = True
                    self.drag_moved = False
                    self.drag_start = (x, y)
                    self.drag_end = (x, y)
            else:
                if self.index_box_rect:
                    left, top, right, bottom = self.index_box_rect
                    if left <= x <= right and top <= y <= bottom:
                        self._jump_to_index_dialog()
                        return

                for action_id, left, top, right, bottom in self.button_rects:
                    if left <= x <= right and top <= y <= bottom:
                        self._handle_button_click(action_id)
                        break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging and self.drag_start:
                diff_x = abs(x - self.drag_start[0])
                diff_y = abs(y - self.drag_start[1])
                if diff_x > DRAG_THRESHOLD or diff_y > DRAG_THRESHOLD:
                    self.drag_moved = True
                    clamped_x = max(
                        self.offset_x, min(x, self.offset_x + self.display_width - 1)
                    )
                    clamped_y = max(
                        self.offset_y, min(y, self.offset_y + self.display_height - 1)
                    )
                    self.drag_end = (clamped_x, clamped_y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_dragging:
                self.is_dragging = False
                if self.drag_moved and self.drag_start and self.drag_end:
                    self._add_box_from_drag(self.drag_start, self.drag_end)
                elif not self.drag_moved and y < panel_top:
                    self._select_box_at(x, y)
                self.drag_start = None
                self.drag_end = None

    def _handle_button_click(self, action_id: str):
        if action_id == "PREV":
            self.stop_auto_play()
            self.prev_image()
        elif action_id == "NEXT":
            self.stop_auto_play()
            self.next_image()
        elif action_id == "AUTO_PLAY":
            self.toggle_auto_play()
        elif action_id == "DEL_BOX":
            self.delete_selected_box()
        elif action_id == "DEL_IMG":
            self.stop_auto_play()
            self.delete_current_image()
        elif action_id == "TOGGLE_LABELS":
            self.toggle_labels()
        elif action_id.startswith("CLASS_"):
            self.set_selected_class(int(action_id.split("_")[1]))

    def stop_auto_play(self):
        if self.is_auto_playing:
            self.is_auto_playing = False
            self.sync_progress()
            self.cached_canvas = None

    def toggle_auto_play(self):
        self.is_auto_playing = not self.is_auto_playing
        if not self.is_auto_playing:
            self.sync_progress()
        else:
            self.prev_auto_play_time = time.perf_counter()
        self.cached_canvas = None

    def process_auto_play(self):
        if not self.is_auto_playing or self.is_dragging:
            return

        if self.current_index >= len(self.image_paths) - 1:
            self.stop_auto_play()
            return

        now = time.perf_counter()
        if now - self.prev_auto_play_time >= AUTO_PLAY_INTERVAL_SEC:
            self.prev_auto_play_time = now
            self.next_image(sync_disk=False)

    def toggle_labels(self):
        self.show_labels = not self.show_labels
        self.cached_canvas = None

    def _select_box_at(self, x: int, y: int):
        hit_index = None
        for index in range(len(self.annotations) - 1, -1, -1):
            left, top, right, bottom = self._annotation_to_rect(self.annotations[index])
            if left <= x <= right and top <= y <= bottom:
                hit_index = index
                break
        self.selected_index = hit_index
        self.cached_canvas = None

    def process_key_hold(self, hwnd=None):
        if (
            self.is_dragging
            or not HAS_WIN32
            or not _is_window_active(hwnd, self.window_name)
        ):
            if self.is_nav_holding:
                self.sync_progress()
                self.is_nav_holding = False
            self.nav_key_down = None
            return

        is_space = bool(ctypes.windll.user32.GetAsyncKeyState(VK_SPACE) & 0x8000)
        is_right = bool(ctypes.windll.user32.GetAsyncKeyState(VK_RIGHT) & 0x8000)
        is_left = bool(ctypes.windll.user32.GetAsyncKeyState(VK_LEFT) & 0x8000)
        is_up = bool(ctypes.windll.user32.GetAsyncKeyState(VK_UP) & 0x8000)

        is_next = is_space or is_right
        is_prev = is_left or is_up

        if is_next and is_prev:
            active_dir = None
        elif is_next:
            active_dir = "next"
        elif is_prev:
            active_dir = "prev"
        else:
            active_dir = None

        if active_dir is not None and self.is_auto_playing:
            self.stop_auto_play()

        now = time.perf_counter()

        if active_dir is None:
            if self.is_nav_holding:
                self.sync_progress()
                self.is_nav_holding = False
            self.nav_key_down = None
            return

        if self.nav_key_down != active_dir:
            self.nav_key_down = active_dir
            self.nav_press_start_time = now
            self.prev_nav_repeat_time = now
            self.is_nav_holding = False
            if active_dir == "next":
                self.next_image()
            else:
                self.prev_image()
        else:
            if now - self.nav_press_start_time >= HOLD_INIT_DELAY_SEC:
                self.is_nav_holding = True
                if now - self.prev_nav_repeat_time >= HOLD_REPEAT_INTERVAL_SEC:
                    self.prev_nav_repeat_time = now
                    if active_dir == "next":
                        self.next_image(sync_disk=False)
                    else:
                        self.prev_image(sync_disk=False)

    def handle_keyboard(self, key: int):
        if key == KEY_ESC:
            self.running = False
        elif not HAS_WIN32 and key in KEYS_ARROW_LEFT:
            self.prev_image()
        elif not HAS_WIN32 and key in (*KEYS_ARROW_RIGHT, KEY_SPACE):
            self.next_image()
        elif key in KEYS_DELETE:
            self.delete_selected_box()


def _mouse_callback(event: int, x: int, y: int, flags: int, reviewer):
    if reviewer:
        reviewer.on_mouse_event(event, x, y, flags, None)


def _get_window_hwnd(title_sub: str):
    found = 0

    def enum_win(hwnd, _):
        nonlocal found
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            if title_sub in buf.value:
                found = hwnd
                return False
        return True

    try:
        cb = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)(enum_win)
        ctypes.windll.user32.EnumWindows(cb, 0)
    except Exception:
        pass
    return found


def _get_window_size(hwnd, window_name: str = ""):
    if hwnd:
        try:
            rect = wintypes.RECT()
            if ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect)):
                w = rect.right - rect.left
                h = rect.bottom - rect.top
                if w > MIN_RESIZE_SIZE[0] and h > MIN_RESIZE_SIZE[1]:
                    return w, h
        except Exception:
            pass
    try:
        rect = cv2.getWindowImageRect(window_name)
        if (
            rect
            and len(rect) == 4
            and rect[2] > MIN_RESIZE_SIZE[0]
            and rect[3] > MIN_RESIZE_SIZE[1]
        ):
            return rect[2], rect[3]
    except Exception:
        pass
    return None


def _is_window_active(hwnd=None, window_name: str = ""):
    if not HAS_WIN32:
        return False
    try:
        fg_hwnd = ctypes.windll.user32.GetForegroundWindow()
        if not fg_hwnd:
            return False
        if hwnd and (
            fg_hwnd == hwnd
            or ctypes.windll.user32.GetAncestor(fg_hwnd, GA_ROOT) == hwnd
        ):
            return True
        length = ctypes.windll.user32.GetWindowTextLengthW(fg_hwnd)
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(fg_hwnd, buf, length + 1)
            return window_name in buf.value if window_name else False
    except Exception:
        pass
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description="Review and edit a YOLO image dataset")
    parser.add_argument(
        "--dataset", type=Path,
        help="Dataset root containing images/ and labels/; omitted to open a folder picker",
    )
    args = parser.parse_args(argv)
    dataset_dir = args.dataset.expanduser().resolve() if args.dataset else choose_dataset_directory()
    if dataset_dir is None:
        return
    if not (dataset_dir / "images").is_dir():
        parser.error(f"Choose the dataset root containing images/ and labels/: {dataset_dir}")
    try:
        reviewer = DatasetReviewer(dataset_dir=dataset_dir, dataset_name=dataset_dir.name)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not reviewer.image_paths:
        parser.error(f"No image files found in '{dataset_dir / 'images'}'.")

    signal.signal(signal.SIGINT, lambda *_: setattr(reviewer, "running", False))

    cv2.namedWindow(reviewer.window_name, cv2.WINDOW_NORMAL)
    init_width, init_height = DatasetReviewer.get_init_window_size()
    cv2.resizeWindow(reviewer.window_name, init_width, init_height)
    reviewer.update_window_size(init_width, init_height)
    cv2.setMouseCallback(reviewer.window_name, _mouse_callback, reviewer)
    reviewer._update_window_title()

    init_canvas = reviewer.get_display_canvas()
    if init_canvas is not None:
        cv2.imshow(reviewer.window_name, init_canvas)
        cv2.waitKey(1)
    hwnd = _get_window_hwnd(reviewer.window_name)

    try:
        while reviewer.running and reviewer.image_paths:
            if not hwnd:
                hwnd = _get_window_hwnd(reviewer.window_name)
            size = _get_window_size(hwnd, reviewer.window_name)
            if size:
                reviewer.update_window_size(size[0], size[1])

            reviewer.process_key_hold(hwnd)
            reviewer.process_auto_play()

            canvas = reviewer.get_display_canvas()
            if canvas is None:
                reviewer.next_image()
                continue

            cv2.imshow(reviewer.window_name, canvas)

            key = cv2.waitKeyEx(LOOP_DELAY_MS)
            if key != -1:
                reviewer.handle_keyboard(key)

            if HAS_WIN32 and hwnd:
                if not ctypes.windll.user32.IsWindow(hwnd):
                    break
            else:
                try:
                    if (
                        cv2.getWindowProperty(
                            reviewer.window_name, cv2.WND_PROP_VISIBLE
                        )
                        == 0
                    ):
                        break
                except Exception:
                    pass

    except KeyboardInterrupt:
        pass
    finally:
        reviewer.stop_auto_play()
        reviewer.sync_progress()
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass


if __name__ == "__main__":
    main()

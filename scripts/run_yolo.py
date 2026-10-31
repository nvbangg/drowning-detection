"""Predict full videos with either or both trained YOLO models."""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "person-in-water": "best-yolo-person-in-water.pt",
    "drowning": "best-yolo-drowning.pt",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "demo/originals")
    parser.add_argument("--model", choices=["both", *MODELS], default="both")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="0, 1 or cpu; default: auto")
    args = parser.parse_args()
    videos = [args.source] if args.source.is_file() else sorted(args.source.glob("*.mp4"))
    if not videos:
        parser.error("No videos found in demo/originals or --source")
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")

    for tag in MODELS if args.model == "both" else [args.model]:
        model = YOLO(str(ROOT / "models" / MODELS[tag]))
        for video in videos:
            output = ROOT / "demo" / f"{video.stem}__{tag}.mp4"
            if output.exists():
                print(f"Skip existing: {output.name}", flush=True)
                continue
            capture = cv2.VideoCapture(str(video))
            fps = capture.get(cv2.CAP_PROP_FPS)
            size = tuple(int(capture.get(key)) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
            if not capture.isOpened() or not 0 < fps < float("inf") or min(size) <= 0:
                capture.release()
                raise ValueError(f"Invalid video: {video}")
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not writer.isOpened():
                capture.release()
                raise OSError(f"Cannot save video: {output}")
            print(f"Predicting {video.name} -> {output.name}", flush=True)
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    result = model.predict(frame, conf=args.conf, device=args.device, verbose=False)[0]
                    writer.write(result.plot(line_width=2))
            except BaseException:
                writer.release()
                output.unlink(missing_ok=True)  # Remove only this interrupted result.
                raise
            finally:
                capture.release()
                writer.release()
            print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()

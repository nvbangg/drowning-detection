"""Download only YOLO Lifeguard sources, then export full-frame 10-FPS temporal clips.

Run: python scripts/download_lifeguard.py
Raw downloads and the provenance CSV stay under raw/lifeguard; demo contains MP4 files only.
"""

import argparse
import csv
import json
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def seconds(value):
    h, m, s = map(float, value.split(":"))
    return h * 3600 + m * 60 + s


def snippet(row):
    # Reconstruct the raw annotation key; crop coordinates identify the annotation only.
    video, start, end, x, y, width = row[:6]
    return (video + "_" + start[:-3].replace(":", "-") + "--" + end[:-3].replace(":", "-")
            + "_" + "--".join(v[:-2].replace(".", "-") for v in (x, y, width)))


def select_clips(dataset, raw):
    assignments = {}
    with ZipFile(dataset) as archive:
        for name in archive.namelist():
            parts = Path(name).parts
            if len(parts) != 3 or parts[0] != "images" or not parts[2].startswith("lifeguard__"):
                continue
            key = Path(parts[2]).stem.removeprefix("lifeguard__").rsplit("__", 1)[0]
            if parts[1] not in {"train", "val"}:
                raise ValueError(f"Unsupported split: {name}")
            if key in assignments and assignments[key] != parts[1]:
                raise ValueError(f"Snippet crosses splits: {key}")
            assignments[key] = parts[1]
    if not assignments:
        raise ValueError("No source-identifiable Lifeguard images found in the YOLO ZIP")
    rows = {}
    with (raw / "datalist.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if row and snippet(row) in assignments:
                key = snippet(row)
                if key in rows and rows[key] != row:
                    raise ValueError(f"Ambiguous raw time mapping: {key}")
                rows[key] = row
    missing = assignments.keys() - rows.keys()
    if missing:
        raise ValueError(f"Missing precise raw time ranges: {sorted(missing)}")
    clips, source_splits = {}, {}
    for key, split in sorted(assignments.items()):
        video, start, end = rows[key][:3]
        if video in source_splits and source_splits[video] != split:
            raise ValueError(f"Source video appears in train and val: {video}")
        source_splits[video] = split
        if seconds(start) < 0 or seconds(end) <= seconds(start):
            raise ValueError(f"Invalid time range: {rows[key]}")
        # Multiple crop annotations of the exact same time range require one full-frame clip.
        clips[(split, video, start, end)] = dict(split=split, video_id=video, start=start, end=end)
    return list(clips.values()), len(assignments)


def probe(path, ffprobe):
    result = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,duration",
                             "-of", "json", str(path)], capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)["streams"][0]
    info["fps"] = float(Fraction(info["avg_frame_rate"]))
    info["duration"] = float(info["duration"])
    return info


def verify(path, original, duration, ffprobe):
    info = probe(path, ffprobe)
    assert info["codec_name"] == "h264" and info["pix_fmt"] == "yuv420p", "Unsupported output codec"
    assert (info["width"], info["height"]) == (original["width"], original["height"]), "Output must retain full frame"
    assert abs(info["fps"] - 10) < 1e-6, "Output is not 10 FPS"
    assert abs(info["duration"] - duration) <= .151, "Wrong temporal cut"
    assert abs(int(info["nb_frames"]) - duration * 10) <= 1.51, "Wrong frame count"
    return info


def download_source(video_id, raw):
    sources = raw / "videos_source"
    candidates = [p for p in sources.iterdir() if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov"}
                  and (p.stem == video_id or p.stem.endswith(f"[{video_id}]"))]
    if candidates:
        return sorted(candidates, key=lambda p: p.suffix.lower() != ".mp4")[0]
    from yt_dlp import YoutubeDL

    options = dict(format="bv*[ext=mp4][vcodec^=avc1][height<=1080]+ba[ext=m4a]/b[ext=mp4][vcodec^=avc1][height<=1080]/bv*[ext=mp4][height<=1080]+ba[ext=m4a]/b[ext=mp4]/best[height<=1080]",
                   outtmpl=str(sources / f"{video_id}.%(ext)s"), merge_output_format="mp4",
                   noplaylist=True, quiet=True, noprogress=True, socket_timeout=20, retries=2,
                   fragment_retries=2, extractor_retries=1, windowsfilenames=True,
                   writesubtitles=False, writethumbnail=False, writeinfojson=False)
    if shutil.which("node"):
        options["js_runtimes"] = {"node": {}}
    print(f"DOWNLOAD {video_id}", flush=True)
    with YoutubeDL(options) as downloader:
        downloader.download([f"https://www.youtube.com/watch?v={video_id}"])
    result = sources / f"{video_id}.mp4"
    if not result.is_file():
        raise FileNotFoundError(f"Download did not produce a complete MP4: {video_id}")
    return result


def prepare_source(video_id, clips, raw, output, ffmpeg, ffprobe):
    results = []
    try:
        source = download_source(video_id, raw)
        info = probe(source, ffprobe)
        if min(info["width"], info["height"]) <= 224:
            raise ValueError("Source is too small; refusing possible 224px training crop")
    except Exception as error:
        for clip in clips:
            results.append(dict(clip, status="error", output="", error=str(error)))
        print(f"SOURCE FAILED {video_id}: {error}", flush=True)
        return results
    for clip in clips:
        start, end, split = clip["start"], clip["end"], clip["split"]
        filename = f"{video_id}__{start.replace(':', '-')}--{end.replace(':', '-')}.mp4"
        target = output / split / filename
        duration = seconds(end) - seconds(start)
        temporary = raw / ".clip-work" / split / filename
        temporary.parent.mkdir(parents=True, exist_ok=True)
        try:
            if seconds(end) > info["duration"] + .15:
                raise ValueError(f"Annotation end {end} exceeds source duration {info['duration']:.3f}s")
            valid = False
            if target.exists():
                try:
                    verify(target, info, duration, ffprobe)
                    valid = True
                except Exception:
                    pass
            if not valid:
                command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-threads", "2",
                           "-ss", start, "-i", str(source), "-t", f"{duration:.6f}", "-map", "0:v:0", "-map", "0:a?",
                           "-vf", "fps=10", "-fps_mode", "cfr", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                           "-pix_fmt", "yuv420p", "-threads", "2", "-c:a", "aac", "-b:a", "128k",
                           "-movflags", "+faststart", str(temporary)]
                run = subprocess.run(command, capture_output=True, text=True)
                if run.returncode:
                    raise RuntimeError(run.stderr[-1200:])
                verify(temporary, info, duration, ffprobe)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary.replace(target)
            results.append(dict(clip, status="ok", output=str(target.relative_to(ROOT)), error=""))
            print(f"READY {split}/{filename}", flush=True)
        except Exception as error:
            results.append(dict(clip, status="error", output="", error=str(error)))
            print(f"CLIP FAILED {video_id} {start}-{end}: {error}", flush=True)
        finally:
            temporary.unlink(missing_ok=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2, choices=range(1, 5))
    parser.add_argument("--plan", action="store_true", help="Print the exact source/time plan without downloading")
    parser.add_argument("--skip", nargs="*", default=[], metavar="VIDEO_ID", help="Source IDs to skip explicitly")
    args = parser.parse_args()
    raw, output = ROOT / "raw/lifeguard", ROOT / "demo/lifeguard"
    clips, annotations = select_clips(ROOT / "datasets/yolo-dataset-drowning.zip", raw)
    grouped = {}
    for clip in clips:
        grouped.setdefault(clip["video_id"], []).append(clip)
    print(f"YOLO: {annotations} crop-annotation segments -> {len(clips)} full-frame clips / {len(grouped)} source videos", flush=True)
    print(dict(Counter(c["split"] for c in clips)), flush=True)
    if args.plan:
        for clip in clips:
            print(clip)
        return
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg and ffprobe must be on PATH")
    (raw / "videos_source").mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = []
        for video_id, entries in sorted(grouped.items()):
            if video_id in args.skip:
                results.extend(dict(clip, status="skipped", output="", error="Explicitly skipped by user") for clip in entries)
                continue
            jobs.append(pool.submit(prepare_source, video_id, entries, raw, output, ffmpeg, ffprobe))
        for job in as_completed(jobs):
            results.extend(job.result())
    report = raw / "demo-clips.csv"
    with report.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "video_id", "start", "end", "status", "output", "error"])
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: (r["split"], r["video_id"], r["start"])))
    failed = [r for r in results if r["status"] == "error"]
    summary = dict(Counter(r["status"] for r in results))
    print(f"DONE {summary}; expected {len(clips)} clips; report: {report}", flush=True)
    if failed:
        raise SystemExit(1)  # Rerun resumes by checking complete outputs and cached source videos.


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

# These imports are only needed for --verify
# (ultralytics will auto-download yolov8n on first use)
try:
    from ultralytics import YOLO
    import cv2
except Exception:
    YOLO = None
    cv2 = None

CONFIG_PATH = Path("./src/detect.config.json")


@dataclass
class DetectConfig:
    # 2.0: scope & schema
    schema_version: str
    classes: list  # COCO class ids for vehicles
    class_map: dict  # {id: "name"}
    bbox_format: str  # "xyxy" in pixel coords
    output_format: str  # "jsonl" (one object per line)
    output_fields: list  # fields in each detection record, 2.0 target

    # 2.1: detector choice & thresholds
    model: str  # e.g., "yolov8n.pt"
    conf: float  # detection confidence threshold
    iou_nms: float  # NMS IoU threshold
    device: str  # "auto" | "cpu" | "cuda"


def default_config() -> DetectConfig:
    return DetectConfig(
        schema_version="1.0",
        classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck (COCO)
        class_map={"2": "car", "3": "motorcycle", "5": "bus", "7": "truck"},
        bbox_format="xyxy",
        output_format="jsonl",
        output_fields=["frame", "t", "obj_id", "bbox", "cls", "score"],
        model="yolov8n.pt",
        conf=0.30,
        iou_nms=0.50,
        device="auto",
    )


def write_config(cfg: DetectConfig, path: Path = CONFIG_PATH):
    with path.open("w") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"Wrote {path}")


def load_config(path: Path = CONFIG_PATH) -> DetectConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}. Run with --init first.")
    data = json.loads(path.read_text())
    return DetectConfig(**data)


def sanity_check_config(cfg: DetectConfig):
    errs = []
    if cfg.bbox_format != "xyxy":
        errs.append("bbox_format must be 'xyxy'.")
    if cfg.output_format.lower() != "jsonl":
        errs.append("output_format should be 'jsonl'.")
    if not all(isinstance(c, int) for c in cfg.classes):
        errs.append("classes must be a list of COCO int ids.")
    if not (0.0 < cfg.conf < 1.0):
        errs.append("conf should be between 0 and 1 (e.g., 0.30).")
    if not (0.0 < cfg.iou_nms <= 1.0):
        errs.append("iou_nms should be in (0,1].")
    if "bbox" not in cfg.output_fields:
        errs.append("output_fields must include 'bbox'.")
    if errs:
        raise ValueError("Config sanity check failed:\n- " + "\n- ".join(errs))


def verify_on_video(cfg: DetectConfig, video_path: Path, sample_frames: int = 5):
    if YOLO is None or cv2 is None:
        raise RuntimeError("ultralytics and opencv-python are required for --verify.")

    # Load video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Basic video sanity
    if W <= 0 or H <= 0 or fps <= 0 or n_frames <= 0:
        raise RuntimeError("Video probe failed (width/height/fps/frames invalid).")

    # Evenly sample a few frames
    indices = sorted(
        set(
            int(i)
            for i in [
                round(k * (n_frames - 1) / max(1, sample_frames - 1))
                for k in range(sample_frames)
            ]
        )
    )

    # Load model with thresholds
    model = YOLO(cfg.model)
    # Note: Ultralytics handles NMS internally; we set conf and classes.
    det_counts = []
    bad_boxes = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {idx}")

        res = model(frame, conf=cfg.conf, classes=cfg.classes, verbose=False)[0]
        # Check boxes
        if res.boxes is None or len(res.boxes) == 0:
            det_counts.append(0)
            continue

        xyxy = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy()

        det_counts.append(len(xyxy))

        # Validate boxes are within image and ordered
        for (x1, y1, x2, y2), s, c in zip(xyxy, scores, cls_ids):
            if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
                bad_boxes += 1
            if s < 0 or s > 1:
                bad_boxes += 1
            if int(c) not in cfg.classes:
                bad_boxes += 1

    cap.release()

    # Simple success criteria
    total = sum(det_counts)
    nonempty_frames = sum(1 for k in det_counts if k > 0)

    report = {
        "video": str(video_path.resolve()),
        "width": W,
        "height": H,
        "fps": fps,
        "frames": n_frames,
        "sampled_frames": indices,
        "detections_per_sampled_frame": det_counts,
        "total_sampled_detections": total,
        "frames_with_at_least_one_detection": nonempty_frames,
        "invalid_boxes_found": bad_boxes,
        "ok": (nonempty_frames >= 2 and bad_boxes == 0),
    }
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        sys.exit(2)


def detect_video(
    cfg: DetectConfig, video_path: Path, output_path: Path, every_n_frames: int = 1
):
    """Run detection on video and output JSONL with one detection per line."""
    if YOLO is None or cv2 is None:
        raise RuntimeError("ultralytics and opencv-python are required for detection.")

    # Load video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if W <= 0 or H <= 0 or fps <= 0 or n_frames <= 0:
        raise RuntimeError("Video probe failed (width/height/fps/frames invalid).")

    # Load model
    model = YOLO(cfg.model)

    # Open output file
    with output_path.open("w") as outf:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process every N frames
            if frame_idx % every_n_frames == 0:
                # Run detection
                results = model(
                    frame, conf=cfg.conf, classes=cfg.classes, verbose=False
                )[0]

                # Calculate timestamp
                timestamp = frame_idx / fps if fps > 0 else 0.0

                # Process detections
                if results.boxes is not None and len(results.boxes) > 0:
                    xyxy = results.boxes.xyxy.cpu().numpy()
                    scores = results.boxes.conf.cpu().numpy()
                    cls_ids = results.boxes.cls.cpu().numpy()

                    for i, (bbox, score, cls_id) in enumerate(
                        zip(xyxy, scores, cls_ids)
                    ):
                        x1, y1, x2, y2 = bbox

                        # Create detection record
                        detection = {
                            "frame": frame_idx,
                            "t": timestamp,
                            "obj_id": None,  # Will be assigned by tracker later
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "cls": int(cls_id),
                            "score": float(score),
                        }

                        # Write to JSONL
                        outf.write(json.dumps(detection, separators=(",", ":")) + "\n")

            frame_idx += 1

    cap.release()
    print(f"Detection complete. Processed {frame_idx} frames, output: {output_path}")


def main():
    p = argparse.ArgumentParser(
        description="Detection config and frame-by-frame detection (Steps 2.0, 2.1, 3.1)."
    )
    p.add_argument(
        "--init",
        action="store_true",
        help="Write detect.config.json with default settings.",
    )
    p.add_argument(
        "--verify",
        type=str,
        metavar="VIDEO",
        help="Verify config on a local video path.",
    )
    p.add_argument(
        "--video",
        type=str,
        metavar="VIDEO",
        help="Run detection on video file.",
    )
    p.add_argument(
        "--config",
        type=str,
        default="src/detect.config.json",
        help="Path to detection config file.",
    )
    p.add_argument(
        "--out",
        type=str,
        metavar="OUTPUT",
        help="Output JSONL file path.",
    )
    p.add_argument(
        "--every",
        type=int,
        default=1,
        help="Process every N frames (default: 1).",
    )
    args = p.parse_args()

    if args.init:
        write_config(default_config())

    if args.verify:
        cfg = load_config(Path(args.config))
        sanity_check_config(cfg)
        verify_on_video(cfg, Path(args.verify))

    if args.video and args.out:
        cfg = load_config(Path(args.config))
        sanity_check_config(cfg)
        detect_video(cfg, Path(args.video), Path(args.out), args.every)

    if not any([args.init, args.verify, (args.video and args.out)]):
        print(
            "Nothing to do. Use --init, --verify <video>, or --video <video> --out <output>",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

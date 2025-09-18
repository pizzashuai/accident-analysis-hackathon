import cv2
from pathlib import Path


def fourcc_to_str(fourcc_float: float) -> str:
    # OpenCV returns FOURCC as float; convert to 4-char code
    i = int(fourcc_float)
    return "".join([chr((i >> 8 * k) & 0xFF) for k in range(4)])


def get_video_info(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    frame_count_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fourcc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))

    # If frame count is unreliable (0/-1), quickly count frames for short clips
    frame_count = frame_count_prop
    if frame_count <= 0:
        cnt = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            cnt += 1
        frame_count = cnt
        cap.release()
        cap = cv2.VideoCapture(str(video_path))  # reopen for any later use

    duration_sec = (frame_count / fps) if fps > 0 else None

    # Grab first/last timestamps if available
    first_ts_ms = None
    last_ts_ms = None
    if frame_count > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _ = cap.read()
        first_ts_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
        _ = cap.read()
        last_ts_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))

    cap.release()

    return {
        "path": str(video_path.resolve()),
        "width": width,
        "height": height,
        "fps": round(fps, 6),
        "frame_count": frame_count,
        "duration_sec": None if duration_sec is None else round(duration_sec, 6),
        "fourcc": fourcc.strip("\x00"),
        "timestamps_ms": {
            "first": None if first_ts_ms is None else round(first_ts_ms, 3),
            "last": None if last_ts_ms is None else round(last_ts_ms, 3),
        },
        "probe_notes": (
            "frame_count from property was 0; counted by scanning"
            if frame_count_prop <= 0
            else "frame_count from property"
        ),
    }


def is_video_valid(video_path: Path) -> dict:
    """Get video info and return it as a dict. Raises exception on error."""
    return get_video_info(video_path)

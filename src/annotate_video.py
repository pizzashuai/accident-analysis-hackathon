import cv2
import supervision as sv
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np

from .persist_detections import (
    read_detections_from_jsonl,
    get_detections_by_frame,
    convert_jsonl_to_supervision_detections,
)


class VideoAnnotator:
    """Handles video annotation with detections from various sources."""

    def __init__(
        self,
        trail_length: int = 10,
        box_color: Optional[sv.Color] = None,
        text_color: Optional[sv.Color] = None,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        box_thickness: int = 2,
    ):
        """
        Initialize video annotator.

        Args:
            trail_length: Length of tracking trails
            box_color: Color for bounding boxes
            text_color: Color for text labels
            text_scale: Scale for text labels
            text_thickness: Thickness for text labels
            box_thickness: Thickness for bounding boxes
        """
        self.trail_length = trail_length
        self.box_color = box_color or sv.Color.WHITE
        self.text_color = text_color or sv.Color.BLACK
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.box_thickness = box_thickness

        # Initialize annotators
        self.box_annotator = sv.BoxAnnotator(
            color=self.box_color, thickness=self.box_thickness
        )
        self.label_annotator = sv.LabelAnnotator(
            color=self.text_color,
            text_scale=self.text_scale,
            text_thickness=self.text_thickness,
        )

        # Tracking history for trails
        self.tracking_history = {}

    def annotate_video_from_detections(
        self,
        video_path: Union[str, Path],
        detections: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None,
        show_trails: bool = True,
        show_labels: bool = True,
        show_boxes: bool = True,
    ) -> Path:
        """
        Annotate video using detection data from JSONL format.

        Args:
            video_path: Path to input video
            detections: List of detection dictionaries
            output_path: Path for output video (default: input_video_annotated.mp4)
            show_trails: Whether to show tracking trails
            show_labels: Whether to show labels
            show_boxes: Whether to show bounding boxes

        Returns:
            Path to the annotated video
        """
        video_path = Path(video_path)

        if output_path is None:
            output_path = video_path.parent / f"{video_path.stem}_annotated.mp4"
        else:
            output_path = Path(output_path)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        print(f"Annotating video: {video_path}")
        print(f"Output: {output_path}")
        print(f"Total frames: {total_frames}")

        frame_count = 0
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get detections for this frame
            frame_detections = get_detections_by_frame(detections, frame_count)

            if frame_detections:
                # Convert to supervision format
                sv_detections = convert_jsonl_to_supervision_detections(
                    frame_detections
                )

                # Annotate frame
                annotated_frame = self._get_annotated_frame(
                    frame, sv_detections, show_trails, show_labels, show_boxes
                )
            else:
                annotated_frame = frame

            # Write frame
            out.write(annotated_frame)
            processed_frames += 1

            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

            frame_count += 1

        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Annotation complete! Processed {processed_frames} frames")
        print(f"Annotated video saved to: {output_path}")

        return output_path

    def annotate_video_from_jsonl(
        self,
        original_video_path: Union[str, Path],
        jsonl_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        show_trails: bool = True,
        show_labels: bool = True,
        show_boxes: bool = True,
    ) -> Path:
        """
        Annotate video using detection data from JSONL file.

        Args:
            video_path: Path to input video
            jsonl_path: Path to JSONL file with detections
            output_path: Path for output video (default: input_video_annotated.mp4)
            show_trails: Whether to show tracking trails
            show_labels: Whether to show labels
            show_boxes: Whether to show bounding boxes

        Returns:
            Path to the annotated video
        """
        # Read detections from JSONL
        detections = read_detections_from_jsonl(Path(jsonl_path))

        return self.annotate_video_from_detections(
            original_video_path,
            detections,
            output_path,
            show_trails,
            show_labels,
            show_boxes,
        )

    def annotate_video_from_supervision_detections(
        self,
        original_video_path: Union[str, Path],
        detections_list: List[sv.Detections],
        output_path: Optional[Union[str, Path]] = None,
        show_trails: bool = True,
        show_labels: bool = True,
        show_boxes: bool = True,
    ) -> Path:
        """
        Annotate video using supervision Detections objects.

        Args:
            original_video_path: Path to input video
            detections_list: List of supervision Detections objects (one per frame)
            output_path: Path for output video (default: input_video_annotated.mp4)
            show_trails: Whether to show tracking trails
            show_labels: Whether to show labels
            show_boxes: Whether to show bounding boxes

        Returns:
            Path to the annotated video
        """
        video_path = Path(original_video_path)

        if output_path is None:
            output_path = video_path.parent / f"{video_path.stem}_annotated.mp4"
        else:
            output_path = Path(output_path)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        print(f"Annotating video: {video_path}")
        print(f"Output: {output_path}")
        print(f"Total frames: {total_frames}")

        frame_count = 0
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get detections for this frame
            if frame_count < len(detections_list):
                detections = detections_list[frame_count]

                # Annotate frame
                annotated_frame = self._get_annotated_frame(
                    frame, detections, show_trails, show_labels, show_boxes
                )
            else:
                annotated_frame = frame

            # Write frame
            out.write(annotated_frame)
            processed_frames += 1

            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

            frame_count += 1

        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Annotation complete! Processed {processed_frames} frames")
        print(f"Annotated video saved to: {output_path}")

        return output_path

    def _get_annotated_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        show_trails: bool,
        show_labels: bool,
        show_boxes: bool,
    ) -> np.ndarray:
        """
        Annotate a single frame with detections.

        Args:
            frame: Input frame
            detections: Supervision detections
            show_trails: Whether to show tracking trails
            show_labels: Whether to show labels
            show_boxes: Whether to show bounding boxes

        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()

        if len(detections) == 0:
            return annotated_frame

        # Update tracking history for trails
        if show_trails and detections.tracker_id is not None:
            self._update_tracking_history(detections)

        # Draw bounding boxes
        if show_boxes:
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

        # Draw labels
        if show_labels:
            labels = self._create_labels(detections)
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

        # Draw trails
        if show_trails:
            annotated_frame = self._draw_trails(annotated_frame)

        return annotated_frame

    def _update_tracking_history(self, detections: sv.Detections) -> None:
        """Update tracking history for trail drawing."""
        if detections.tracker_id is None:
            return

        for detection_idx in range(len(detections)):
            tracker_id = detections.tracker_id[detection_idx]
            if tracker_id is not None:
                if tracker_id not in self.tracking_history:
                    self.tracking_history[tracker_id] = []

                # Get center point of bounding box
                x1, y1, x2, y2 = detections.xyxy[detection_idx]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                self.tracking_history[tracker_id].append((center_x, center_y))

                # Keep only the last trail_length points
                if len(self.tracking_history[tracker_id]) > self.trail_length:
                    self.tracking_history[tracker_id] = self.tracking_history[
                        tracker_id
                    ][-self.trail_length :]

    def _create_labels(self, detections: sv.Detections) -> List[str]:
        """Create labels for detections."""
        labels = []
        if detections.tracker_id is not None:
            for detection_idx in range(len(detections)):
                tracker_id = detections.tracker_id[detection_idx]
                if tracker_id is not None:
                    labels.append(f"ID:{tracker_id}")
                else:
                    labels.append("")
        else:
            labels = [""] * len(detections)
        return labels

    def _draw_trails(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracking trails on frame."""
        for tracker_id, trace_points in self.tracking_history.items():
            if len(trace_points) > 1:
                # Get color for this tracker ID
                color = sv.ColorPalette.DEFAULT.by_idx(tracker_id % 20)
                color_bgr = (int(color.b), int(color.g), int(color.r))

                # Draw lines connecting the trail points
                for i in range(1, len(trace_points)):
                    pt1 = trace_points[i - 1]
                    pt2 = trace_points[i]
                    cv2.line(frame, pt1, pt2, color_bgr, 2)

        return frame


def annotate_video(
    video_path: Union[str, Path],
    detections_source: Union[str, Path, List[Dict[str, Any]], List[sv.Detections]],
    output_path: Optional[Union[str, Path]] = None,
    trail_length: int = 10,
    show_trails: bool = True,
    show_labels: bool = True,
    show_boxes: bool = True,
) -> Path:
    """
    Convenience function to annotate video with detections.

    Args:
        video_path: Path to input video
        detections_source: Source of detections (JSONL path, detection list, or supervision detections)
        output_path: Path for output video (default: input_video_annotated.mp4)
        trail_length: Length of tracking trails
        show_trails: Whether to show tracking trails
        show_labels: Whether to show labels
        show_boxes: Whether to show bounding boxes

    Returns:
        Path to the annotated video
    """
    annotator = VideoAnnotator(trail_length=trail_length)

    # Determine detections source type
    if isinstance(detections_source, (str, Path)):
        # Assume it's a JSONL file path
        return annotator.annotate_video_from_jsonl(
            video_path,
            detections_source,
            output_path,
            show_trails,
            show_labels,
            show_boxes,
        )
    elif (
        isinstance(detections_source, list)
        and len(detections_source) > 0
        and detections_source[0] is not None
    ):
        if isinstance(detections_source[0], dict):
            # List of detection dictionaries
            from typing import cast

            detections_dict_list = cast(List[Dict[str, Any]], detections_source)
            return annotator.annotate_video_from_detections(
                video_path,
                detections_dict_list,
                output_path,
                show_trails,
                show_labels,
                show_boxes,
            )
        elif isinstance(detections_source[0], sv.Detections):
            # List of supervision Detections
            from typing import cast

            detections_sv_list = cast(List[sv.Detections], detections_source)
            return annotator.annotate_video_from_supervision_detections(
                video_path,
                detections_sv_list,
                output_path,
                show_trails,
                show_labels,
                show_boxes,
            )

    raise ValueError("Invalid detections source type")

Tools

- load_detections(track_ids, frame_range=None, fields=None, require_pairing=True, fps_hint=None)
  Returns paired rows sorted by frame with metadata (frame, timestamp, bbox_xyxy, world_coords, speed_mph, heading_deg, flags
  about missing data). Allow track_ids to be 1–N so the agent can pull any subset; frame_range lets it restrict load; fields
  controls payload size; require_pairing=False gives single-track data when the agent wants context; include summary stats (fps,
  timestamps seen, missing frames).
- compute_pair_metrics(pairs, iou_threshold=0.01, vehicle_length_m=4.5, include_headings=False)
  Accepts list of paired detections (ideally the output of load_detections). Optional thresholds keep it flexible;
  include_headings toggles heading vector math. Return list of enriched rows with IoU, pixel distance, great-circle distance,
  relative speed, plus boolean flags for threshold crossings.
- trace_impact_window(metric_rows, iou_threshold=0.01, distance_threshold_m=5.0, persistence_frames=3)
  Finds first qualifying frame, last frame still overlapping, min-distance frame, and collects surrounding indices. Include
  collision_detected flag, impact_frames, closest_approach, overlap_duration_frames, and diagnostic notes (e.g., “threshold hit
  only by IoU”).
- build_timeline(metric_rows, impact_summary, padding_frames=10, stages=None)
  Produces structured timeline entries with stage, frame, timestamp, excerpts of metrics, and brief narrative text. stages lets
  the LLM supply custom phase labels (defaults to [approach, first_contact, peak_overlap, separation]). Include token-efficient
  summary table ready for final response.
- report_assumptions(metric_rows, metadata, warn_if_missing=("world_coords","speed_mph"))
  Scans the data for anomalies or missing values; warn_if_missing is configurable so the agent can emphasize the metrics it
  intends to cite. Return short strings describing each assumption/gap.

• Metric Rows Explained

- Each metric row is the enriched record returned by compute_pair_metrics for one frame (or timestamp) where both tracks appear. Think
  of it as the raw paired detections plus all derived quantities the downstream tools need.
- Recommended fields:
  - frame, timestamp, and track_ids (the pair included)
  - bbox_xyxy: original bounding boxes per track; optionally pixel_center_distance_px
  - world_coords for both tracks, with world_distance_m (great-circle or planar distance)
  - iou (intersection-over-union), relative_speed_mps (or mph), and optional heading_diff_deg
  - Derived flags like iou_exceeds_threshold, distance_below_threshold, collision_candidate
  - Provenance metadata: filters applied, data gaps, fps/time-delta hints

Usage

- trace_impact_window scans metric_rows to detect first contact, minimum distance, and overlap duration.
- build_timeline pulls the same rows to quote key metrics per stage.
- report_assumptions inspects them for missing speeds, inconsistent coordinates, etc.

Keep the structure consistent (e.g., a list of dictionaries) so the LLM can index by frame, subset by flags, or compute additional
summaries if needed.

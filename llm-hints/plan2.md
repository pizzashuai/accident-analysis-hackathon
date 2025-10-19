Given the example jsonl backend/src/common/features/process_video/detections.jsonl, I need the following logic inside backend/src/common/features/postprocess

• Token-Efficient Collision Plan

- Preprocess: parse the JSONL once, filter to the track IDs of interest, sort by frame/time, and confirm both tracks appear at
  the same timestamps (skip rows without the pair). Record fps or time delta if available.
- Per-frame metrics: for every paired frame, compute (1) IoU of the two bbox_xyxy values, (2) euclidean pixel-center distance,
  (3) great-circle distance using world_coords, (4) relative speed from speed_mph (sum when moving head-on), and (5) any heading/
  velocity vector if the data has it.
- Collision detection: flag the first frame where IoU exceeds a small threshold (e.g. 0.01) or world-distance falls below a
  physical threshold (≈ car length). Track how long the overlap persists, plus the frame with minimum distance—these become the
  core “impact window”.
- Summaries for the LLM: instead of the raw JSONL, feed a compact table with only the key frames: first overlap, peak overlap,
  minimum distance, and a few frames before/after. Each row should include frame, time, IoU, world distance (m), pixel distance,
  speeds, and averaged collision location (lat, lon).
- Context checks: verify sensor/coordinate consistency (e.g. world_coords realistic, speeds non-null) and note any gaps or
  interpolation you perform—LLM needs to know assumptions.
- LLM prompt skeleton: send a short description of the scenario, the derived metrics table, and a request like “produce a
  timeline of events (approach, first contact, peak collision, post-impact) using the table above; cite frame/time and metrics in
  each step.”

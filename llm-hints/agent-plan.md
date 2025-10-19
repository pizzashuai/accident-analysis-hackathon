• Agent Blueprint

- Establish context in the system prompt: describe the collision-analysis objective, note that backend/src/common/features/
  process_video/detections.jsonl is the raw source, and emphasize accuracy, unit checking, and citing frames/timestamps in
  outputs. Include an explicit instruction that the agent must decide which analytical tool to run and in what order.
- Provide the assistant message with a concise recap of available resources (tool list, schema overview of JSONL rows, definition
  of key metrics like IoU, great-circle distance, speed_mph). Mention FPS/time-delta assumptions if stored alongside detections;
  require the agent to surface any missing metadata as warnings.

Tooling Plan

Implemented in backend/src/common/features/postprocess/tools.py

Orchestration Flow

- Agent receives user goal (“produce an event timeline from the detection JSONL”). First action: call load_detections with the
  two track IDs the user specifies (or ask for them if not provided). If metadata like fps is absent, agent records that via
  report_assumptions.
- Next, invoke compute_pair_metrics on the paired frames; inspect results for sanity (non-decreasing timestamps, plausible
  distances).
- Use trace_impact_window to isolate the critical frames. If no collision threshold is met, agent should branch and explain that
  only near-miss data exists but still describe the closest approach.
- Call build_timeline to get structured timeline entries; merge with caveats from report_assumptions.
- Compose final answer: short scenario recap, chronological bullet list citing frame/time/metrics, highlight impact window,
  mention remaining uncertainties.

LLM Prompt Skeleton

- System: “You are an accident reconstruction analyst… ensure every event cites frame/time and key numeric metrics… request a
  follow-up tool call whenever more data is needed.”
- User goal template: “Use available tools to parse detections, compute per-frame proximity metrics, determine if/when a
  collision occurs, and produce a timeline covering approach → first contact → peak impact → post-impact.”
- Planning reminder: require the agent to outline its intended sequence (e.g., “Will load detections, then compute metrics, then
  evaluate collision window…”) before invoking tools, so you can trace decisions.
- Response format: instruct the LLM to return (1) bullet timeline with citations (frame, timestamp, IoU, world distance, relative
  speed), (2) concise impact summary, (3) assumptions/uncertainties section.

Validation & Iteration

- After implementing tools, test them with a small known slice of detections.jsonl to ensure pairing/filtering works. Capture
  sample outputs so the agent prompt can reference expected field names.
- Run at least one dry run where you simulate the agent calls manually to confirm the prompt drives the correct sequence.
- Add logging inside each tool to surface unexpected data issues; ensure messages are short so the agent transcript stays token-
  efficient.
- Future extension: optionally add a plotting tool (e.g., to graph distance vs. time) that the agent can call when textual
  description isn’t sufficient.

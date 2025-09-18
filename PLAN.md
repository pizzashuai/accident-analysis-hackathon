# Traffic Accident Causal Timeline — TODO Runbook

I'm using Python uv to manage deps. Use ./.venv/ Python env to run the code

## Phase 1 — Video Probe

### \[x] 1.1 Parse video info

**Do:** run your CLI (from earlier)

```bash
python main.py data/accident.mp4 --pretty > out/video_info.json
```

**Verify:**

```bash
jq -r '.fps,.frame_count,.duration_sec' out/video_info.json
# Expect: fps>0, frame_count>0, duration≈3s
```

---

## Phase 2 — Detection Config (2.0 & 2.1)

### \[x] 2.1 Generate defaults

**Do:**

```bash
python detect_config.py --init
```

**Verify:**

```bash
jq -r '.model,.classes|length,.conf,.iou_nms' detect.config.json
# Expect: yolov8n.pt, 4, ~0.3, ~0.5
```

### \[x] 2.2 Light probe on sample frames

**Do:**

```bash
python detect_config.py --verify data/accident.mp4 > out/detect_verify.json
```

**Verify:**

```bash
jq -r '.ok, .frames_with_at_least_one_detection, .invalid_boxes_found' out/detect_verify.json
# Expect: true, >=2, 0
```

---

## Phase 3 — Detect per Frame (no tracking yet)

### \[x] 3.1 Frame-by-frame detect → JSONL (schema only)

**Do:** (your detect script)

```bash
python detect.py --video data/accident.mp4 --config detect.config.json --out out/dets.jsonl --every 1
```

**Verify:**

```bash
# schema + monotonic time
python - <<'PY'
import json,sys
W=H=None
prev_t=-1
ok=1
for i,l in enumerate(open("out/dets.jsonl")):
    r=json.loads(l)
    if not (isinstance(r.get("bbox"),list) and len(r["bbox"])==4): ok=0;break
    if r["t"]<prev_t: ok=0;break
    x1,y1,x2,y2=r["bbox"]
    if not (0<=x1<x2 and 0<=y1<y2): ok=0;break
    prev_t=r["t"]
print("ok" if ok else "bad"); sys.exit(0 if ok else 2)
PY
```

---

## Phase 4 — Tracking (ByteTrack / OC/BoT-SORT)

### \[x] 4.1 Run tracker

**Do:**

```bash
python track.py --video data/accident.mp4 --config detect.config.json \
  --tracker bytetrack.yaml --out out/tracks.jsonl
```

**Verify:**

```bash
# basic stats
python - <<'PY'
import json,collections
ids=set(); frames=collections.Counter()
for l in open("out/tracks.jsonl"):
    r=json.loads(l); ids.add(r["obj_id"]); frames[r["obj_id"]]+=1
print("unique_ids",len(ids),"avg_len",sum(frames.values())/max(1,len(ids)))
PY
# Expect: unique_ids>=1, avg_len >= 0.3*FPS (approx)
```

---

## Phase 5 — Visual Overlay (sanity)

### \[ ] 5.1 Render overlay with IDs + trails

**Do:**

```bash
python overlay.py --video data/accident.mp4 --tracks out/tracks.jsonl --out out/overlay.mp4 --trail 10
```

**Verify:** visually inspect

```bash
# Confirm video opens and IDs are stable across frames
```

---

## Phase 6 — Smooth & Fill Tiny Gaps

### \[ ] 6.1 EMA/Savitzky–Golay on centers

**Do:**

```bash
python smooth.py --tracks out/tracks.jsonl --out out/tracks_smooth.jsonl --ema 0.5 --gap 2
```

**Verify:**

```bash
# no NaNs, same or longer per-ID lengths
python - <<'PY'
import json,collections,math
def stats(p):
  frames=collections.Counter()
  for l in open(p):
    r=json.loads(l); frames[r["obj_id"]]+=1
  return frames
a=stats("out/tracks.jsonl"); b=stats("out/tracks_smooth.jsonl")
ok=all(b[k]>=a.get(k,0) for k in b)
print("ok" if ok else "bad")
PY
```

---

## Phase 7 — Kinematics (per-frame vx, vy, speed, heading)

### \[ ] 7.1 Derive velocity & speed

**Do:**

```bash
python kinematics.py --tracks out/tracks_smooth.jsonl --fps $(jq -r '.fps' out/video_info.json) \
  --out out/kinematics.jsonl
```

**Verify:**

```bash
python - <<'PY'
import json,math,sys
ok=1
for i,l in enumerate(open("out/kinematics.jsonl")):
    r=json.loads(l)
    if any(math.isnan(r[k]) or math.isinf(r[k]) for k in ("vx","vy","speed")): ok=0;break
    if r["speed"]<0: ok=0;break
print("ok" if ok else "bad"); sys.exit(0 if ok else 2)
PY
```

---

## Phase 8 — Contact / Collision Candidates

### \[ ] 8.1 IoU / distance contact frames

**Do:**

```bash
python collisions.py --tracks out/tracks_smooth.jsonl --iou 0.05 --dist 30 \
  --out out/contacts.jsonl
```

**Verify:**

```bash
# either some contacts detected or an explicit flag
if test -s out/contacts.jsonl; then echo "contacts found"; else echo "no_impact_detected"; fi
```

---

## Phase 9 — Event Primitives

### \[ ] 9.1 Emit events (entered, hard_brake, impact, stopped)

**Do:**

```bash
python events.py --kin out/kinematics.jsonl --contacts out/contacts.jsonl \
  --hard_brake_drop 0.4 --stop_eps 5 --stop_ms 300 \
  --out out/events.json
```

**Verify:**

```bash
jq -r 'type, (.[0].type // "none")' out/events.json
# Expect: "array" and at least one event type if visible
```

---

## Phase 10 — Minimal Timeline

### \[ ] 10.1 Compose per-object timelines

**Do:**

```bash
python timeline.py --events out/events.json --out out/timeline.json
```

**Verify:**

```bash
jq -r '.[0].obj_id, (.[0].steps|length)' out/timeline.json
# Expect: valid obj_id and >=1 step
```

---

## Phase 11 — Quick Metrics

### \[ ] 11.1 Pre/post-impact speed & Δv

**Do:**

```bash
python metrics.py --kin out/kinematics.jsonl --events out/events.json \
  --window_ms 100 --out out/metrics.json
```

**Verify:**

```bash
jq -r '.[] | "\(.obj_id) \(.delta_v)"' out/metrics.json
# Expect: non-negative numbers
```

---

## Phase 12 — Manual Patch (optional)

### \[ ] 12.1 Remap IDs over a frame range

**Do:**

```bash
python id_patch.py --tracks out/tracks.jsonl --from 7 --to 3 --start 45 --end 60 \
  --out out/tracks_patched.jsonl
```

**Verify:**

```bash
# IDs no longer flip across the patched range; re-run smoothing/events as needed
```

---

## Phase 13 — One-shot Orchestration (smoke test)

### \[ ] 13.1 Single entrypoint

**Do:**

```bash
python run.py --video data/accident.mp4 --out out --tracker bytetrack.yaml --conf 0.25
```

**Verify:**

```bash
test -s out/tracks.jsonl && test -s out/overlay.mp4 && test -s out/events.json && echo "pipeline ok"
```

---

## Success Criteria (for your 3-second clip)

- ≥1 stable track lasting ≥1.0 s
- If collision visible: at least one `impact` event + Δv computed
- `overlay.mp4` clearly shows IDs and impact moment

---

**Tip:** Keep each script tiny and single-responsibility so these verifies stay fast and deterministic.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { MouseEvent } from 'react';
import {
  Badge,
  Box,
  Button,
  Card,
  Checkbox,
  Divider,
  FileInput,
  Group,
  ScrollArea,
  Stack,
  Switch,
  Text,
} from '@mantine/core';
import { IconEraser, IconEye, IconEyeOff, IconUpload } from '@tabler/icons-react';

type BboxTuple = [number, number, number, number];

export interface DetectionRecord {
  video_id?: string;
  frame: number;
  time: number;
  track_id: number | null;
  det_idx?: number;
  class_id?: number;
  class_name?: string;
  conf?: number;
  bbox_xyxy: BboxTuple;
  center?: [number, number];
}

interface VideoAnnotationViewerProps {
  videoUrl: string;
  /**
   * Optional detections the parent already resolved (e.g., from API).
   * When omitted a local JSONL upload is required.
   */
  initialDetections?: DetectionRecord[];
}

interface TrackSummary {
  trackId: number;
  frameCount: number;
  firstSeen: number;
  classes: string[];
  maxConfidence?: number;
}

const TIME_KEY_SCALE = 1000;

const parseJsonlDetections = (payload: string): DetectionRecord[] => {
  const lines = payload.split(/\r?\n/);
  const parsed: DetectionRecord[] = [];

  lines.forEach((raw, index) => {
    const line = raw.trim();
    if (!line) {
      return;
    }

    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch (error) {
      throw new Error(`Line ${index + 1}: ${String(error)}`);
    }

    if (typeof value !== 'object' || value === null) {
      throw new Error(`Line ${index + 1}: expected JSON object`);
    }

    const obj = value as Record<string, unknown>;

    const bbox = obj.bbox_xyxy;
    if (
      !Array.isArray(bbox) ||
      bbox.length !== 4 ||
      bbox.some((point) => typeof point !== 'number')
    ) {
      throw new Error(`Line ${index + 1}: bbox_xyxy must be an array of four numbers`);
    }

    const frame = Number(obj.frame);
    const time = Number(obj.time);
    const trackIdValue = obj.track_id;
    const trackId =
      trackIdValue === null || trackIdValue === undefined
        ? null
        : Number(trackIdValue);

    if (!Number.isFinite(frame) || frame < 0) {
      throw new Error(`Line ${index + 1}: frame must be a non-negative number`);
    }

    if (!Number.isFinite(time) || time < 0) {
      throw new Error(`Line ${index + 1}: time must be a non-negative number`);
    }

    if (trackId !== null && !Number.isFinite(trackId)) {
      throw new Error(`Line ${index + 1}: track_id must be numeric or null`);
    }

    const record: DetectionRecord = {
      video_id: typeof obj.video_id === 'string' ? obj.video_id : undefined,
      frame,
      time,
      track_id: trackId,
      det_idx: typeof obj.det_idx === 'number' ? obj.det_idx : undefined,
      class_id: typeof obj.class_id === 'number' ? obj.class_id : undefined,
      class_name:
        typeof obj.class_name === 'string' ? obj.class_name : undefined,
      conf:
        obj.conf === undefined
          ? undefined
          : Number.isFinite(Number(obj.conf))
            ? Number(obj.conf)
            : undefined,
      bbox_xyxy: bbox as BboxTuple,
      center: Array.isArray(obj.center)
        ? (obj.center.slice(0, 2) as [number, number])
        : undefined,
    };

    parsed.push(record);
  });

  return parsed;
};

const buildDetectionTimeIndex = (detections: DetectionRecord[]) => {
  const index = new Map<number, DetectionRecord[]>();

  detections.forEach((det) => {
    if (!det || !det.bbox_xyxy) {
      return;
    }
    const key = Math.round(det.time * TIME_KEY_SCALE);
    const entry = index.get(key);
    if (entry) {
      entry.push(det);
    } else {
      index.set(key, [det]);
    }
  });

  return index;
};

const extractTrackSummaries = (detections: DetectionRecord[]): TrackSummary[] => {
  const map = new Map<number, TrackSummary>();

  detections.forEach((det) => {
    if (det.track_id === null || det.track_id === undefined) {
      return;
    }

    const existing = map.get(det.track_id);
    if (existing) {
      existing.frameCount += 1;
      existing.firstSeen = Math.min(existing.firstSeen, det.time);
      if (det.class_name && !existing.classes.includes(det.class_name)) {
        existing.classes.push(det.class_name);
      }
      if (
        det.conf !== undefined &&
        (existing.maxConfidence === undefined || det.conf > existing.maxConfidence)
      ) {
        existing.maxConfidence = det.conf;
      }
    } else {
      map.set(det.track_id, {
        trackId: det.track_id,
        frameCount: 1,
        firstSeen: det.time,
        classes: det.class_name ? [det.class_name] : [],
        maxConfidence: det.conf,
      });
    }
  });

  return Array.from(map.values()).sort((a, b) => a.trackId - b.trackId);
};

const colorForTrack = (trackId: number): string => {
  const hue = (trackId * 47) % 360;
  return `hsl(${hue}, 85%, 60%)`;
};

const formatDetectionLabel = (det: DetectionRecord): string => {
  const parts: string[] = [];
  if (det.track_id !== null && det.track_id !== undefined) {
    parts.push(`Track ${det.track_id}`);
  }
  if (det.class_name) {
    parts.push(det.class_name);
  }
  if (det.conf !== undefined && Number.isFinite(det.conf)) {
    parts.push(`${Math.round(det.conf * 100)}%`);
  }
  return parts.join(' · ');
};

export const VideoAnnotationViewer = ({
  videoUrl,
  initialDetections = [],
}: VideoAnnotationViewerProps) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationHandle = useRef<number | null>(null);
  const currentFrameDetectionsRef = useRef<DetectionRecord[]>([]);

  const [detections, setDetections] = useState<DetectionRecord[]>(initialDetections);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);
  const [enabledTrackIds, setEnabledTrackIds] = useState<number[]>([]);
  const [showUntracked, setShowUntracked] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [focusedTrackId, setFocusedTrackId] = useState<number | null>(null);

  const detectionIndex = useMemo(
    () => buildDetectionTimeIndex(detections),
    [detections],
  );

  const trackSummaries = useMemo(
    () => extractTrackSummaries(detections),
    [detections],
  );

  useEffect(() => {
    if (!initialDetections.length) {
      return;
    }
    setDetections(initialDetections);
    setEnabledTrackIds((ids) =>
      ids.length ? ids : extractTrackSummaries(initialDetections).map((t) => t.trackId),
    );
  }, [initialDetections]);

  useEffect(() => {
    const trackIds = trackSummaries.map((track) => track.trackId);
    setEnabledTrackIds((prev) => {
      if (!prev.length) {
        return trackIds;
      }
      const next = prev.filter((trackId) => trackIds.includes(trackId));
      return next;
    });
  }, [trackSummaries]);

  const enabledTrackSet = useMemo(
    () => new Set(enabledTrackIds),
    [enabledTrackIds],
  );

  const handleFileChange = useCallback(
    (file: File | null) => {
      setParseError(null);
      setSelectedFile(file);

      if (!file) {
        setDetections(initialDetections);
        return;
      }

      const reader = new FileReader();

      reader.onload = () => {
        try {
          const payload = typeof reader.result === 'string' ? reader.result : '';
          const parsed = parseJsonlDetections(payload);
          setDetections(parsed);
          setEnabledTrackIds(extractTrackSummaries(parsed).map((t) => t.trackId));
        } catch (error) {
          setParseError(error instanceof Error ? error.message : String(error));
          setDetections(initialDetections);
        }
      };

      reader.onerror = () => {
        setParseError('Failed to read JSONL file.');
        setDetections(initialDetections);
      };

      reader.readAsText(file);
    },
    [initialDetections],
  );

  const toggleTrack = useCallback((trackId: number) => {
    setEnabledTrackIds((prev) => {
      if (prev.includes(trackId)) {
        return prev.filter((id) => id !== trackId);
      }
      return [...prev, trackId];
    });
  }, []);

  const clearSelections = useCallback(() => {
    setEnabledTrackIds([]);
  }, []);

  const selectAllTracks = useCallback(() => {
    setEnabledTrackIds(trackSummaries.map((track) => track.trackId));
  }, [trackSummaries]);

  const cancelAnimation = useCallback(() => {
    if (animationHandle.current !== null) {
      cancelAnimationFrame(animationHandle.current);
      animationHandle.current = null;
    }
  }, []);

  const drawFrame = useCallback(() => {
    const videoEl = videoRef.current;
    const canvasEl = canvasRef.current;

    if (!videoEl || !canvasEl) {
      return;
    }

    const width = videoEl.videoWidth;
    const height = videoEl.videoHeight;

    if (!width || !height) {
      return;
    }

    if (canvasEl.width !== width || canvasEl.height !== height) {
      canvasEl.width = width;
      canvasEl.height = height;
    }

    const ctx = canvasEl.getContext('2d');
    if (!ctx) {
      return;
    }

    ctx.clearRect(0, 0, width, height);

    const currentKey = Math.round(videoEl.currentTime * TIME_KEY_SCALE);
    const frameDetections = detectionIndex.get(currentKey) ?? [];

    const visibleDetections = frameDetections.filter((det) => {
      if (det.track_id === null || det.track_id === undefined) {
        return showUntracked;
      }
      return enabledTrackSet.has(det.track_id);
    });

    currentFrameDetectionsRef.current = visibleDetections;

    visibleDetections.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox_xyxy;
      const color =
        det.track_id !== null && det.track_id !== undefined
          ? colorForTrack(det.track_id)
          : '#ffffff';

      ctx.strokeStyle = color;
      ctx.lineWidth = det.track_id === focusedTrackId ? 3.5 : 2;
      ctx.shadowColor = 'rgba(0, 0, 0, 0.35)';
      ctx.shadowBlur = 4;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      if (showLabels) {
        const label = formatDetectionLabel(det);
        if (label) {
          ctx.font = '14px Inter, sans-serif';
          ctx.textBaseline = 'top';
          ctx.shadowBlur = 0;
          const textPadding = 4;
          const textWidth = ctx.measureText(label).width + textPadding * 2;
          const textHeight = 18;
          const boxX = x1;
          const boxY = Math.max(0, y1 - textHeight - 4);

          ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
          ctx.fillRect(boxX, boxY, textWidth, textHeight);
          ctx.fillStyle = 'white';
          ctx.fillText(label, boxX + textPadding, boxY + 2);
        }
      }
    });
  }, [
    detectionIndex,
    enabledTrackSet,
    focusedTrackId,
    showLabels,
    showUntracked,
  ]);

  useEffect(() => {
    drawFrame();
  }, [drawFrame]);

  useEffect(() => {
    const videoEl = videoRef.current;
    if (!videoEl) {
      return;
    }

    const handlePlay = () => {
      cancelAnimation();
      const step = () => {
        drawFrame();
        animationHandle.current = requestAnimationFrame(step);
      };
      step();
    };

    const handlePause = () => {
      cancelAnimation();
      drawFrame();
    };

    const handleSeeked = () => {
      drawFrame();
    };

    const handleLoadedMetadata = () => {
      drawFrame();
    };

    const handleTimeUpdate = () => {
      drawFrame();
    };

    videoEl.addEventListener('play', handlePlay);
    videoEl.addEventListener('pause', handlePause);
    videoEl.addEventListener('seeked', handleSeeked);
    videoEl.addEventListener('loadedmetadata', handleLoadedMetadata);
    videoEl.addEventListener('timeupdate', handleTimeUpdate);

    return () => {
      videoEl.removeEventListener('play', handlePlay);
      videoEl.removeEventListener('pause', handlePause);
      videoEl.removeEventListener('seeked', handleSeeked);
      videoEl.removeEventListener('loadedmetadata', handleLoadedMetadata);
      videoEl.removeEventListener('timeupdate', handleTimeUpdate);
      cancelAnimation();
    };
  }, [cancelAnimation, drawFrame]);

  const handleOverlayClick = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      const container = containerRef.current;
      const videoEl = videoRef.current;
      if (!container || !videoEl) {
        return;
      }

      const rect = container.getBoundingClientRect();
      const clickX = event.clientX - rect.left;
      const clickY = event.clientY - rect.top;

      const scaleX = videoEl.videoWidth / rect.width;
      const scaleY = videoEl.videoHeight / rect.height;

      const videoSpaceX = clickX * scaleX;
      const videoSpaceY = clickY * scaleY;

      const hitDetection = currentFrameDetectionsRef.current.find((det) => {
        const [x1, y1, x2, y2] = det.bbox_xyxy;
        return (
          videoSpaceX >= x1 &&
          videoSpaceX <= x2 &&
          videoSpaceY >= y1 &&
          videoSpaceY <= y2
        );
      });

      if (hitDetection && hitDetection.track_id !== null && hitDetection.track_id !== undefined) {
        toggleTrack(hitDetection.track_id);
        setFocusedTrackId(hitDetection.track_id);
      }
    },
    [toggleTrack],
  );

  const activeTrackCount = enabledTrackIds.length;
  const totalTrackCount = trackSummaries.length;

  return (
    <Card withBorder shadow='sm'>
      <Stack gap='md'>
        <Stack gap={4}>
          <Text fw={600}>Video Annotation</Text>
          <Text size='sm' c='dimmed'>
            Overlay detections from JSONL (as produced by backend/src/common/features/process-video) onto the project video. Toggle tracks to focus
            on relevant vehicles without rendering a new video file.
          </Text>
        </Stack>

        <Group align='flex-end'>
          <FileInput
            label='Load detection JSONL'
            placeholder='Upload detections.jsonl'
            accept='.jsonl,.json,.txt'
            value={selectedFile}
            onChange={handleFileChange}
            leftSection={<IconUpload size={16} />}
            withAsterisk={!detections.length}
            clearable
          />
          <Button variant='light' onClick={selectAllTracks} disabled={!totalTrackCount}>
            Select all
          </Button>
          <Button
            variant='subtle'
            color='gray'
            leftSection={<IconEraser size={16} />}
            onClick={clearSelections}
            disabled={!enabledTrackIds.length}
          >
            Clear
          </Button>
        </Group>

        {parseError && (
          <Card withBorder p='sm' radius='md' bg='rgba(255,0,0,0.05)'>
            <Text size='sm' c='red'>
              {parseError}
            </Text>
          </Card>
        )}

        <Group gap='md' wrap='wrap'>
          <Badge color='blue' variant='light'>
            Tracks loaded: {totalTrackCount}
          </Badge>
          <Badge color='green' variant='light'>
            Active tracks: {activeTrackCount}
          </Badge>
          <Badge color='gray' variant='light'>
            Detections: {detections.length}
          </Badge>
        </Group>

        <Group gap='lg' align='flex-start'>
          <Box
            ref={containerRef}
            onClick={handleOverlayClick}
            style={{
              position: 'relative',
              width: '100%',
              maxWidth: '720px',
              cursor: 'crosshair',
            }}
          >
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              style={{ width: '100%', display: 'block' }}
            >
              Your browser does not support the video tag.
            </video>
            <canvas
              ref={canvasRef}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
              }}
            />
          </Box>

          <Card withBorder p='sm' style={{ flex: 1, minWidth: '220px' }}>
            <Stack gap='sm'>
              <Group gap='xs'>
                <Switch
                  size='sm'
                  color='blue'
                  onLabel={<IconEye size={14} />}
                  offLabel={<IconEyeOff size={14} />}
                  checked={showLabels}
                  onChange={(event) => setShowLabels(event.currentTarget.checked)}
                  label='Show labels'
                />
                <Switch
                  size='sm'
                  checked={showUntracked}
                  onChange={(event) => setShowUntracked(event.currentTarget.checked)}
                  label='Show untracked detections'
                />
              </Group>

              <Divider label='Tracks' />
              {totalTrackCount === 0 ? (
                <Text size='sm' c='dimmed'>
                  Load a JSONL file with detections to view tracks.
                </Text>
              ) : (
                <ScrollArea h={260} type='auto'>
                  <Stack gap='xs'>
                    {trackSummaries.map((track) => {
                      const checked = enabledTrackSet.has(track.trackId);
                      const color = colorForTrack(track.trackId);
                      return (
                        <Checkbox
                          key={track.trackId}
                          label={
                            <Stack gap={2}>
                              <Group gap='xs'>
                                <Badge
                                  color='gray'
                                  variant='light'
                                  style={{
                                    color: '#1f2933',
                                    backgroundColor: `${color}33`,
                                    border: `1px solid ${color}`,
                                  }}
                                >
                                  Track {track.trackId}
                                </Badge>
                                <Text size='xs' c='dimmed'>
                                  {track.frameCount} frames · starts at {track.firstSeen.toFixed(2)}s
                                </Text>
                              </Group>
                              {track.classes.length > 0 && (
                                <Text size='xs' c='dimmed'>
                                  Classes: {track.classes.join(', ')}
                                </Text>
                              )}
                            </Stack>
                          }
                          checked={checked}
                          onChange={() => toggleTrack(track.trackId)}
                          onMouseEnter={() => setFocusedTrackId(track.trackId)}
                          onMouseLeave={() => setFocusedTrackId(null)}
                        />
                      );
                    })}
                  </Stack>
                </ScrollArea>
              )}
            </Stack>
          </Card>
        </Group>
      </Stack>
    </Card>
  );
};

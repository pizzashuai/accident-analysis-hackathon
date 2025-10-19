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
  Loader,
  Alert,
} from '@mantine/core';
import {
  IconEraser,
  IconEye,
  IconEyeOff,
  IconUpload,
  IconDownload,
  IconFilter,
} from '@tabler/icons-react';
import {
  useDetections,
  useArtifactDownloadUrl,
  useArtifactContent,
  useProcessingRuns,
} from '../../hooks/useProcessing';
import { JsonlFileSelector } from './JsonlFileSelector';
import { VideoMapAnimation } from './VideoMapAnimation';

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
  speed_mph?: number;
  world_coords?: [number, number];
}

interface VideoAnnotationViewerProps {
  videoUrl: string;
  /**
   * Optional detections the parent already resolved (e.g., from API).
   * When omitted a local JSONL upload is required.
   */
  initialDetections?: DetectionRecord[];
  /**
   * Optional run ID for API-based detections
   */
  runId?: string;
  /**
   * Optional project ID for JSONL artifact selection
   */
  projectId?: string;
}

interface TrackSummary {
  trackId: number;
  frameCount: number;
  firstSeen: number;
  classes: string[];
  maxConfidence?: number;
}

const TIME_KEY_SCALE = 1000; // Convert seconds to milliseconds for indexing

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
      throw new Error(
        `Line ${index + 1}: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
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
      throw new Error(
        `Line ${index + 1}: bbox_xyxy must be an array of four numbers`
      );
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
      speed_mph:
        obj.speed_mph === undefined
          ? undefined
          : Number.isFinite(Number(obj.speed_mph))
            ? Number(obj.speed_mph)
            : undefined,
      world_coords:
        Array.isArray(obj.world_coords) && obj.world_coords.length >= 2
          ? (obj.world_coords.slice(0, 2) as [number, number])
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

const extractTrackSummaries = (
  detections: DetectionRecord[]
): TrackSummary[] => {
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
        (existing.maxConfidence === undefined ||
          det.conf > existing.maxConfidence)
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
  if (det.speed_mph !== undefined && Number.isFinite(det.speed_mph)) {
    parts.push(`${det.speed_mph.toFixed(1)} mph`);
  }
  return parts.join(' · ');
};

export const VideoAnnotationViewer = ({
  videoUrl,
  initialDetections = [],
  runId,
  projectId,
}: VideoAnnotationViewerProps) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationHandle = useRef<number | null>(null);
  const currentFrameDetectionsRef = useRef<DetectionRecord[]>([]);

  const [detections, setDetections] =
    useState<DetectionRecord[]>(initialDetections);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedArtifactId, setSelectedArtifactId] = useState<string | null>(
    null
  );
  const [useApiDetections, setUseApiDetections] = useState<boolean>(false);
  const [parseError, setParseError] = useState<string | null>(null);
  const [enabledTrackIds, setEnabledTrackIds] = useState<number[]>([]);
  const [showUntracked, setShowUntracked] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [focusedTrackId, setFocusedTrackId] = useState<number | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isFiltering, setIsFiltering] = useState(false);
  const [filterError, setFilterError] = useState<string | null>(null);
  const [filterSuccess, setFilterSuccess] = useState<string | null>(null);

  // Calculate initial map configuration from detections
  const initialMapConfig = useMemo(() => {
    // Calculate from first batch of detections with world_coords
    const withCoords = detections.filter((d) => d.world_coords);
    if (withCoords.length === 0) {
      return { center: { lat: 1.3521, lng: 103.8198 }, zoom: 20 };
    }

    const lats = withCoords.map((d) => d.world_coords![1]);
    const lngs = withCoords.map((d) => d.world_coords![0]);
    const centerLat = (Math.min(...lats) + Math.max(...lats)) / 2;
    const centerLng = (Math.min(...lngs) + Math.max(...lngs)) / 2;

    return {
      center: { lat: centerLat, lng: centerLng },
      zoom: 19, // Higher zoom level for larger map view
    };
  }, [detections.length]); // Only recalculate when detections change
  const [currentFrameDetections, setCurrentFrameDetections] = useState<
    DetectionRecord[]
  >([]);

  // Throttled state for map updates to reduce re-rendering
  const [mapDetections, setMapDetections] = useState<DetectionRecord[]>([]);
  const [currentFrameNumber, setCurrentFrameNumber] = useState<number>(0);
  const lastMapUpdateTime = useRef<number>(0);
  const MAP_UPDATE_INTERVAL = 250; // Update map 4 times per second

  // Get available processing runs for API detection selection
  const { data: processingRuns } = useProcessingRuns(
    projectId || '',
    !!projectId
  );

  // Determine which runId to use for API detections
  const effectiveRunId = useApiDetections
    ? runId ||
      processingRuns?.data?.find((run: any) => run.status === 'completed')?.id
    : null;

  // API-based detections
  const { data: apiDetections, isLoading: loadingDetections } = useDetections(
    effectiveRunId || '',
    undefined,
    0,
    10000 // Get all detections
  );

  // Artifact-based JSONL loading
  const { data: artifactContent, isLoading: loadingArtifactContent } =
    useArtifactContent(selectedArtifactId || '', !!selectedArtifactId);

  // Convert API detections to DetectionRecord format
  const convertedApiDetections = useMemo((): DetectionRecord[] => {
    if (!apiDetections?.data) return [];

    const converted: DetectionRecord[] = apiDetections.data.map(
      (detection) => ({
        video_id: 'api',
        frame: detection.frame_idx,
        time: detection.t_ms / 1000.0, // Convert milliseconds to seconds
        track_id: detection.track_id,
        class_id:
          typeof detection.extra?.class_id === 'number'
            ? detection.extra.class_id
            : undefined,
        class_name: detection.cls,
        conf: detection.conf ?? undefined,
        bbox_xyxy: [
          detection.x,
          detection.y,
          detection.x + detection.w,
          detection.y + detection.h,
        ] as BboxTuple,
        center:
          Array.isArray(detection.extra?.center) &&
          detection.extra.center.length >= 2
            ? ([detection.extra.center[0], detection.extra.center[1]] as [
                number,
                number,
              ])
            : [detection.x + detection.w / 2, detection.y + detection.h / 2],
        speed_mph:
          typeof detection.extra?.speed_mph === 'number'
            ? detection.extra.speed_mph
            : undefined,
        world_coords:
          detection.wx && detection.wy
            ? [detection.wx, detection.wy]
            : undefined,
      })
    );

    return converted;
  }, [apiDetections]);

  // Load JSONL data from artifact when selected
  useEffect(() => {
    console.log('useEffect artifactContent:', {
      selectedArtifactId,
      artifactContent: !!artifactContent,
    });
    if (!selectedArtifactId || !artifactContent) {
      return;
    }

    const loadArtifactJsonl = async () => {
      try {
        setParseError(null);
        const parsed = parseJsonlDetections(artifactContent as string);
        console.log(
          'Loading artifact JSONL, parsed detections:',
          parsed.length
        );
        setDetections(parsed);
        setEnabledTrackIds(extractTrackSummaries(parsed).map((t) => t.trackId));
        setSelectedFile(null); // Clear file selection when using artifact
      } catch (error) {
        console.error('Error parsing artifact JSONL:', error);
        setParseError(error instanceof Error ? error.message : 'Unknown error');
        setDetections(initialDetections);
      }
    };

    loadArtifactJsonl();
  }, [selectedArtifactId, artifactContent]);

  // Use API detections if useApiDetections is true, otherwise use initialDetections, uploaded file, or artifact
  const effectiveDetections = useApiDetections
    ? convertedApiDetections
    : detections;

  const detectionIndex = useMemo(
    () => buildDetectionTimeIndex(effectiveDetections),
    [effectiveDetections]
  );

  const trackSummaries = useMemo(
    () => extractTrackSummaries(effectiveDetections),
    [effectiveDetections]
  );

  useEffect(() => {
    console.log('useEffect API detections:', {
      initialDetectionsLength: initialDetections.length,
      useApiDetections,
      effectiveDetectionsLength: effectiveDetections.length,
    });
    if (!initialDetections.length && !useApiDetections) {
      return;
    }
    // Only update detections if we're switching to API mode or if we have initial detections
    if (useApiDetections && !initialDetections.length) {
      console.log('Setting detections from API:', effectiveDetections.length);
      setDetections(effectiveDetections);
      setEnabledTrackIds((ids) =>
        ids.length
          ? ids
          : extractTrackSummaries(effectiveDetections).map((t) => t.trackId)
      );
    }
  }, [useApiDetections, initialDetections.length]);

  useEffect(() => {
    const trackIds = trackSummaries.map((track) => track.trackId);
    setEnabledTrackIds((prev) => {
      if (!prev.length) {
        return trackIds;
      }
      const next = prev.filter((trackId) => trackIds.includes(trackId));

      // Only update if the arrays are actually different
      if (
        next.length !== prev.length ||
        !next.every((id, index) => id === prev[index])
      ) {
        return next;
      }
      return prev; // No change needed
    });
  }, [trackSummaries]);

  const enabledTrackSet = useMemo(
    () => new Set(enabledTrackIds),
    [enabledTrackIds]
  );

  const handleFileChange = useCallback(
    (file: File | null) => {
      if (useApiDetections) {
        // Don't allow file upload when using API detections
        return;
      }

      setParseError(null);
      setSelectedFile(file);
      setSelectedArtifactId(null); // Clear artifact selection when file is uploaded
      setUseApiDetections(false); // Clear API detection mode when file is uploaded

      if (!file) {
        setDetections(initialDetections);
        return;
      }

      const reader = new FileReader();

      reader.onload = () => {
        try {
          const payload =
            typeof reader.result === 'string' ? reader.result : '';
          const parsed = parseJsonlDetections(payload);
          setDetections(parsed);
          setEnabledTrackIds(
            extractTrackSummaries(parsed).map((t) => t.trackId)
          );
        } catch (error) {
          setParseError(
            error instanceof Error ? error.message : 'Unknown error'
          );
          setDetections(initialDetections);
        }
      };

      reader.onerror = () => {
        setParseError('Failed to read JSONL file.');
        setDetections(initialDetections);
      };

      reader.readAsText(file);
    },
    [initialDetections, runId]
  );

  const toggleTrack = useCallback((trackId: number) => {
    setEnabledTrackIds((prev) => {
      if (prev.includes(trackId)) {
        return prev.filter((id) => id !== trackId);
      }
      return [...prev, trackId];
    });
    // Clear filter messages when track selection changes
    setFilterError(null);
    setFilterSuccess(null);
  }, []);

  const clearSelections = useCallback(() => {
    setEnabledTrackIds([]);
    setFilterError(null);
    setFilterSuccess(null);
  }, []);

  const selectAllTracks = useCallback(() => {
    setEnabledTrackIds(trackSummaries.map((track) => track.trackId));
    setFilterError(null);
    setFilterSuccess(null);
  }, [trackSummaries]);

  const handleArtifactSelect = useCallback(
    (artifactId: string | null) => {
      setSelectedArtifactId(artifactId);
      setUseApiDetections(false); // Clear API detection mode when artifact is selected
      if (!artifactId) {
        setDetections(initialDetections);
        setSelectedFile(null);
      }
    },
    [initialDetections]
  );

  const handleApiDetectionToggle = useCallback(() => {
    setUseApiDetections(!useApiDetections);
    if (!useApiDetections) {
      // Switching to API mode - clear other selections
      setSelectedArtifactId(null);
      setSelectedFile(null);
    }
  }, [useApiDetections]);

  const handleFilterAndSave = useCallback(async () => {
    if (!enabledTrackIds.length) {
      setFilterError('Please select at least one track to filter');
      return;
    }

    // Determine which artifact to use for filtering
    let artifactIdToUse: string | null = null;

    if (useApiDetections && effectiveRunId) {
      // For API detections, we need to find the JSONL artifact from the run
      // This would require additional API calls to get the run's artifacts
      setFilterError(
        'Filtering API detections is not yet supported. Please use a JSONL file.'
      );
      return;
    } else if (selectedArtifactId) {
      artifactIdToUse = selectedArtifactId;
    } else if (selectedFile) {
      setFilterError(
        'Please use an artifact JSONL file for filtering. Uploaded files are not supported.'
      );
      return;
    } else {
      setFilterError('No JSONL file available for filtering');
      return;
    }

    if (!artifactIdToUse) {
      setFilterError('No valid artifact found for filtering');
      return;
    }

    setIsFiltering(true);
    setFilterError(null);
    setFilterSuccess(null);

    try {
      const { VideoAnnotationService } = await import('../../client/sdk.gen');

      const response = await VideoAnnotationService.filterDetectionsByTracks({
        requestBody: {
          track_ids: enabledTrackIds,
          artifact_id: artifactIdToUse,
          filename: `filtered_tracks_${enabledTrackIds.join('_')}.jsonl`,
        },
      });

      setFilterSuccess(
        `Successfully filtered ${response.detection_count} detections for ${response.track_count} track(s). ` +
          `The original file has been replaced with the filtered version.`
      );
    } catch (error) {
      console.error('Error filtering detections:', error);
      setFilterError(
        error instanceof Error ? error.message : 'Failed to filter detections'
      );
    } finally {
      setIsFiltering(false);
    }
  }, [
    enabledTrackIds,
    useApiDetections,
    effectiveRunId,
    selectedArtifactId,
    selectedFile,
  ]);

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

    // Try exact match first, then look for nearby times (±100ms tolerance)
    let frameDetections = detectionIndex.get(currentKey) ?? [];

    if (frameDetections.length === 0) {
      // Look for detections within ±100ms tolerance
      const tolerance = 100; // 100ms tolerance
      for (let offset = 0; offset <= tolerance; offset += 10) {
        const key1 = currentKey + offset;
        const key2 = currentKey - offset;

        if (detectionIndex.has(key1)) {
          frameDetections = detectionIndex.get(key1) ?? [];
          break;
        }
        if (detectionIndex.has(key2)) {
          frameDetections = detectionIndex.get(key2) ?? [];
          break;
        }
      }
    }

    const visibleDetections = frameDetections.filter((det) => {
      if (det.track_id === null || det.track_id === undefined) {
        return showUntracked;
      }
      return enabledTrackSet.has(det.track_id);
    });

    currentFrameDetectionsRef.current = visibleDetections;
    setCurrentFrameDetections(visibleDetections);

    // Update current frame number from detections
    if (frameDetections.length > 0) {
      const frameNumber = frameDetections[0].frame;
      setCurrentFrameNumber(frameNumber);
    }

    // Throttle map updates to reduce re-rendering
    const now = Date.now();
    if (now - lastMapUpdateTime.current >= MAP_UPDATE_INTERVAL) {
      setMapDetections(visibleDetections);
      lastMapUpdateTime.current = now;
    }

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

      if (
        hitDetection &&
        hitDetection.track_id !== null &&
        hitDetection.track_id !== undefined
      ) {
        toggleTrack(hitDetection.track_id);
        setFocusedTrackId(hitDetection.track_id);
      }
    },
    [toggleTrack]
  );

  const activeTrackCount = enabledTrackIds.length;
  const totalTrackCount = trackSummaries.length;

  return (
    <Card withBorder shadow='sm'>
      <Stack gap='md'>
        <Stack gap={4}>
          <Text fw={600}>Video Annotation</Text>
          <Text size='sm' c='dimmed'>
            {runId
              ? 'Live detection overlay from processed video analysis. Toggle tracks to focus on relevant vehicles.'
              : 'Overlay detections from JSONL. Toggle tracks to focus on relevant vehicles without rendering a new video file.'}
          </Text>
        </Stack>

        <Stack gap='md'>
          {/* Data Source Selection */}
          {projectId &&
            processingRuns?.data?.some(
              (run: any) => run.status === 'completed'
            ) && (
              <Group gap='md' align='flex-end'>
                {useApiDetections && loadingDetections && (
                  <Group gap='xs'>
                    <Loader size='sm' />
                    <Text size='sm' c='dimmed'>
                      Loading API detections...
                    </Text>
                  </Group>
                )}
                {!useApiDetections &&
                  selectedArtifactId &&
                  loadingArtifactContent && (
                    <Group gap='xs'>
                      <Loader size='sm' />
                      <Text size='sm' c='dimmed'>
                        Loading JSONL file...
                      </Text>
                    </Group>
                  )}
              </Group>
            )}

          {/* JSONL Artifact Selection */}
          {projectId && !useApiDetections && (
            <JsonlFileSelector
              projectId={projectId}
              selectedArtifactId={selectedArtifactId}
              onArtifactSelect={handleArtifactSelect}
              disabled={useApiDetections}
            />
          )}

          <Group align='flex-end'>
            {!useApiDetections && !selectedArtifactId && (
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
            )}
            <Button
              variant='light'
              onClick={selectAllTracks}
              disabled={!totalTrackCount}
            >
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
            <Button
              variant='filled'
              color='blue'
              leftSection={<IconFilter size={16} />}
              onClick={handleFilterAndSave}
              disabled={!enabledTrackIds.length || isFiltering}
              loading={isFiltering}
            >
              Filter & Replace File
            </Button>
          </Group>
        </Stack>

        {parseError && (
          <Card withBorder p='sm' radius='md' bg='rgba(255,0,0,0.05)'>
            <Text size='sm' c='red'>
              {parseError}
            </Text>
          </Card>
        )}

        {filterError && (
          <Alert color='red' icon={<IconFilter size={16} />}>
            <Text size='sm'>{filterError}</Text>
          </Alert>
        )}

        {filterSuccess && (
          <Alert color='green' icon={<IconDownload size={16} />}>
            <Text size='sm'>{filterSuccess}</Text>
          </Alert>
        )}

        <Group gap='md' wrap='wrap'>
          <Badge color='blue' variant='light'>
            Tracks loaded: {totalTrackCount}
          </Badge>
          <Badge color='green' variant='light'>
            Active tracks: {activeTrackCount}
          </Badge>
          <Badge color='gray' variant='light'>
            Detections: {effectiveDetections.length}
          </Badge>
          {useApiDetections && (
            <Badge color='purple' variant='light'>
              API Data
            </Badge>
          )}
          {selectedArtifactId && !useApiDetections && (
            <Badge color='orange' variant='light'>
              Artifact JSONL
            </Badge>
          )}
          {selectedFile && !useApiDetections && !selectedArtifactId && (
            <Badge color='cyan' variant='light'>
              Uploaded File
            </Badge>
          )}
        </Group>

        <Stack gap='md'>
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
          </Group>

          {/* Map Animation */}
          <VideoMapAnimation
            detections={mapDetections}
            allDetections={detections}
            currentFrame={currentFrameNumber}
            height={400}
            center={initialMapConfig.center}
            zoom={initialMapConfig.zoom}
            lockView={true}
          />

          {/* Labels and Tracks Display */}
          <Card withBorder p='sm'>
            <Stack gap='sm'>
              <Group gap='xs'>
                <Switch
                  size='sm'
                  color='blue'
                  onLabel={<IconEye size={14} />}
                  offLabel={<IconEyeOff size={14} />}
                  checked={showLabels}
                  onChange={(event) =>
                    setShowLabels(event.currentTarget.checked)
                  }
                  label='Show labels'
                />
                <Switch
                  size='sm'
                  checked={showUntracked}
                  onChange={(event) =>
                    setShowUntracked(event.currentTarget.checked)
                  }
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
                                  {track.frameCount} frames · starts at{' '}
                                  {track.firstSeen.toFixed(2)}s
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
        </Stack>
      </Stack>
    </Card>
  );
};

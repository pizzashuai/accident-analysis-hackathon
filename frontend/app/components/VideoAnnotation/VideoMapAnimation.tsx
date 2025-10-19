import { useEffect, useState, useMemo } from 'react';
import { APIProvider, Map, AdvancedMarker } from '@vis.gl/react-google-maps';
import type { MapCameraChangedEvent } from '@vis.gl/react-google-maps';
import { Paper, Text, Alert, Group, Badge } from '@mantine/core';
import { IconCar, IconMapPin } from '@tabler/icons-react';
import type { DetectionRecord } from './VideoAnnotationViewer';

interface VideoMapAnimationProps {
  detections: DetectionRecord[]; // Current frame detections with world_coords
  height?: number; // Map height (default 400px)
  center?: { lat: number; lng: number }; // Initial center
  zoom?: number; // Initial zoom level
  lockView?: boolean; // Lock zoom and center, disable auto-centering
}

// Reuse the same color function from VideoAnnotationViewer
const colorForTrack = (trackId: number): string => {
  const hue = (trackId * 47) % 360;
  return `hsl(${hue}, 85%, 60%)`;
};

// Custom car icon component
const CarIcon = ({
  trackId,
  speed,
  color,
}: {
  trackId: number;
  speed?: number;
  color: string;
}) => {
  return (
    <div
      style={{
        width: 32,
        height: 32,
        backgroundColor: color,
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        border: '2px solid white',
        boxShadow: '0 2px 4px rgba(0,0,0,0.3)',
        position: 'relative',
      }}
    >
      <IconCar size={18} color='white' />
      <div
        style={{
          position: 'absolute',
          bottom: -20,
          left: '50%',
          transform: 'translateX(-50%)',
          fontSize: '10px',
          fontWeight: 'bold',
          color: '#333',
          backgroundColor: 'rgba(255,255,255,0.9)',
          padding: '2px 4px',
          borderRadius: '4px',
          whiteSpace: 'nowrap',
        }}
      >
        {trackId}
        {speed && ` (${speed.toFixed(1)} mph)`}
      </div>
    </div>
  );
};

export const VideoMapAnimation = ({
  detections,
  height = 400,
  center = { lat: 1.3521, lng: 103.8198 }, // Default to Singapore
  zoom = 20, // Higher default zoom level
  lockView = false,
}: VideoMapAnimationProps) => {
  // Use state for unlocked view, props directly for locked view
  const [mapCenter, setMapCenter] = useState(center);
  const [mapZoom, setMapZoom] = useState(zoom);
  const [hasUserInteracted, setHasUserInteracted] = useState(false);

  // Use props directly when locked, state when unlocked
  const currentCenter = lockView ? center : mapCenter;
  const currentZoom = lockView ? zoom : mapZoom;

  const apiKey = import.meta.env.VITE_GOOGLE_MAP_KEY;

  // Filter detections that have valid world_coords
  const detectionsWithCoords = useMemo(() => {
    return detections.filter(
      (det) =>
        det.world_coords &&
        Array.isArray(det.world_coords) &&
        det.world_coords.length >= 2 &&
        typeof det.world_coords[0] === 'number' &&
        typeof det.world_coords[1] === 'number' &&
        !isNaN(det.world_coords[0]) &&
        !isNaN(det.world_coords[1])
    );
  }, [detections]);

  // Calculate map bounds and center based on visible detections (only when not locked)
  useEffect(() => {
    if (lockView) return; // Skip auto-centering when locked

    if (detectionsWithCoords.length > 0 && !hasUserInteracted) {
      const lats = detectionsWithCoords.map((det) => det.world_coords![1]);
      const lngs = detectionsWithCoords.map((det) => det.world_coords![0]);

      const minLat = Math.min(...lats);
      const maxLat = Math.max(...lats);
      const minLng = Math.min(...lngs);
      const maxLng = Math.max(...lngs);

      // Center the map on the bounds
      const centerLat = (minLat + maxLat) / 2;
      const centerLng = (minLng + maxLng) / 2;

      setMapCenter({ lat: centerLat, lng: centerLng });

      // Adjust zoom based on bounds
      const latDiff = maxLat - minLat;
      const lngDiff = maxLng - minLng;
      const maxDiff = Math.max(latDiff, lngDiff);

      if (maxDiff > 0) {
        let newZoom = 15;
        if (maxDiff > 0.01) newZoom = 12;
        else if (maxDiff > 0.005) newZoom = 14;
        else if (maxDiff > 0.001) newZoom = 16;
        else newZoom = 18;

        setMapZoom(newZoom);
      }
    } else if (detectionsWithCoords.length === 0) {
      // Reset to default when no detections
      setMapCenter(center);
      setMapZoom(zoom);
    }
  }, [detectionsWithCoords, center, zoom, hasUserInteracted, lockView]);

  const handleCameraChanged = (ev: MapCameraChangedEvent) => {
    if (lockView) return; // Ignore camera changes when locked
    setHasUserInteracted(true);
    setMapCenter(ev.detail.center);
    setMapZoom(ev.detail.zoom);
  };

  // Check if any detections have world_coords
  const hasWorldCoords = detections.some(
    (det) =>
      det.world_coords &&
      Array.isArray(det.world_coords) &&
      det.world_coords.length >= 2
  );

  // Memoize markers to prevent unnecessary re-creation
  const markers = useMemo(() => {
    return detectionsWithCoords.map((detection) => {
      if (!detection.world_coords || !detection.track_id) return null;

      const [lng, lat] = detection.world_coords;
      const color = colorForTrack(detection.track_id);

      return (
        <AdvancedMarker
          key={detection.track_id} // Use only track_id for stable keys
          position={{ lat, lng }}
          title={`Track ${detection.track_id}${detection.speed_mph ? ` - ${detection.speed_mph.toFixed(1)} mph` : ''}`}
        >
          <CarIcon
            trackId={detection.track_id}
            speed={detection.speed_mph}
            color={color}
          />
        </AdvancedMarker>
      );
    });
  }, [detectionsWithCoords]);

  // Show message if no world_coords data available
  if (!hasWorldCoords) {
    return (
      <Paper withBorder p='md' style={{ height }}>
        <Alert
          color='yellow'
          icon={<IconMapPin size={16} />}
          title='Map Animation Unavailable'
        >
          <Text size='sm'>
            Map animation requires world coordinate data. Please ensure
            homography is configured and video processing included coordinate
            transformation.
          </Text>
        </Alert>
      </Paper>
    );
  }

  // Show message if no tracks with coords are visible
  if (detectionsWithCoords.length === 0) {
    return (
      <Paper withBorder p='md' style={{ height }}>
        <Alert
          color='blue'
          icon={<IconCar size={16} />}
          title='No Vehicles Visible'
        >
          <Text size='sm'>
            No tracks selected with location data. Enable tracks in the sidebar
            to see vehicles on the map.
          </Text>
        </Alert>
      </Paper>
    );
  }

  return (
    <Paper
      withBorder
      style={{ height, position: 'relative', overflow: 'hidden' }}
    >
      <Group
        justify='space-between'
        p='xs'
        style={{ position: 'absolute', top: 0, left: 0, right: 0, zIndex: 10 }}
      >
        <Badge color='blue' variant='light'>
          Vehicles: {detectionsWithCoords.length}
        </Badge>
        <Badge color='green' variant='light'>
          Live Map
        </Badge>
      </Group>

      <APIProvider apiKey={apiKey}>
        <Map
          zoom={currentZoom}
          center={currentCenter}
          onCameraChanged={lockView ? undefined : handleCameraChanged}
          mapId='video-map-animation'
          style={{ width: '100%', height: '100%' }}
          gestureHandling={lockView ? 'none' : 'auto'}
          disableDefaultUI={true}
        >
          {markers}
        </Map>
      </APIProvider>
    </Paper>
  );
};

export default VideoMapAnimation;

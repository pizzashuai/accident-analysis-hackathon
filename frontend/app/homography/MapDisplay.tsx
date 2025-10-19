import {
  APIProvider,
  Map,
  AdvancedMarker,
  Pin,
} from '@vis.gl/react-google-maps';
import type {
  MapCameraChangedEvent,
  MapMouseEvent,
} from '@vis.gl/react-google-maps';
import { Paper, Button, Group } from '@mantine/core';
import { useEffect, useState } from 'react';
import { IconRefresh } from '@tabler/icons-react';

interface LatLngPoint {
  lat: number;
  lng: number;
}

interface Point {
  xNorm: number;
  yNorm: number;
  x?: number;
  y?: number;
}

interface PointPair {
  id: string | number;
  a: Point;
  b: LatLngPoint;
}

interface MapDisplayProps {
  width?: string | number;
  height?: string | number;
  center?: { lat: number; lng: number };
  zoom?: number;
  apiKey?: string;
  onMapClick?: (latLng: LatLngPoint) => void;
  onMapHover?: (latLng: LatLngPoint) => void;
  pairs?: PointPair[];
  hoveredPairId?: string | number | null;
  pickingMode?: boolean;
  pendingPointA?: Point | null;
}

export function MapDisplay({
  width = '100%',
  height = 400,
  center = { lat: 1.3521, lng: 103.8198 }, // Default to Singapore
  zoom = 12,
  apiKey = import.meta.env.VITE_GOOGLE_MAP_KEY,
  onMapClick,
  onMapHover,
  pairs = [],
  hoveredPairId = null,
  pickingMode = false,
  pendingPointA = null,
}: MapDisplayProps) {
  const [mapCenter, setMapCenter] = useState(center);
  const [mapZoom, setMapZoom] = useState(zoom);
  const [hasUserInteracted, setHasUserInteracted] = useState(false);

  // Update map center when pairs are added or center prop changes
  useEffect(() => {
    if (pairs.length > 0 && !hasUserInteracted) {
      // Calculate bounds of all pairs
      const lats = pairs.map((pair) => pair.b.lat);
      const lngs = pairs.map((pair) => pair.b.lng);
      const minLat = Math.min(...lats);
      const maxLat = Math.max(...lats);
      const minLng = Math.min(...lngs);
      const maxLng = Math.max(...lngs);

      // Center the map on the bounds
      const centerLat = (minLat + maxLat) / 2;
      const centerLng = (minLng + maxLng) / 2;

      setMapCenter({ lat: centerLat, lng: centerLng });

      // Adjust zoom based on bounds only if user hasn't interacted
      const latDiff = maxLat - minLat;
      const lngDiff = maxLng - minLng;
      const maxDiff = Math.max(latDiff, lngDiff);

      if (maxDiff > 0) {
        // Calculate appropriate zoom level
        let newZoom = 15;
        if (maxDiff > 0.01) newZoom = 12;
        else if (maxDiff > 0.005) newZoom = 14;
        else if (maxDiff > 0.001) newZoom = 16;
        else newZoom = 18;

        setMapZoom(newZoom);
      }
    } else if (pairs.length === 0) {
      // Use the provided center and zoom when no pairs
      setMapCenter(center);
      setMapZoom(zoom);
    }
  }, [pairs, center, zoom, hasUserInteracted]);
  const handleMapClick = (event: MapMouseEvent) => {
    if (onMapClick && event.detail.latLng) {
      onMapClick({
        lat: event.detail.latLng.lat,
        lng: event.detail.latLng.lng,
      });
    }
  };

  const handleMapMouseMove = (event: MapMouseEvent) => {
    if (onMapHover && event.detail.latLng && pickingMode) {
      onMapHover({
        lat: event.detail.latLng.lat,
        lng: event.detail.latLng.lng,
      });
    }
  };

  const handleCameraChanged = (ev: MapCameraChangedEvent) => {
    // Mark that user has interacted with the map
    setHasUserInteracted(true);

    // Update local state to reflect user's camera changes
    setMapCenter(ev.detail.center);
    setMapZoom(ev.detail.zoom);
  };

  const resetMapView = () => {
    setHasUserInteracted(false);
    // This will trigger the useEffect to recalculate bounds
  };

  return (
    <div>
      {pairs.length > 0 && (
        <Group justify='flex-end' mb='xs'>
          <Button
            size='xs'
            variant='light'
            leftSection={<IconRefresh size={12} />}
            onClick={resetMapView}
          >
            Reset View
          </Button>
        </Group>
      )}
      <Paper
        withBorder
        style={{
          width,
          height,
          position: 'relative',
          overflow: 'hidden',
          cursor: pickingMode && pendingPointA ? 'crosshair' : 'default',
        }}
      >
        <APIProvider apiKey={apiKey}>
          <Map
            zoom={mapZoom}
            center={mapCenter}
            onCameraChanged={handleCameraChanged}
            onClick={handleMapClick}
            onMousemove={handleMapMouseMove}
            mapId='homography-map'
            style={{ width: '100%', height: '100%' }}
            gestureHandling={pickingMode ? 'greedy' : 'auto'}
            disableDefaultUI={false}
          >
            {/* Render markers for paired points */}
            {pairs.map((pair, index) => {
              const markerKey = String(pair.id);
              const isHovered =
                hoveredPairId !== null &&
                String(hoveredPairId) === markerKey;

              return (
                <AdvancedMarker
                  key={markerKey}
                  position={pair.b}
                  title={`Point ${index + 1}`}
                >
                  <Pin
                    background={isHovered ? '#ff6b6b' : '#4dabf7'}
                    borderColor='#ffffff'
                    glyphColor='#ffffff'
                  >
                    <div
                      style={{
                        fontSize: '14px',
                        fontWeight: 'bold',
                        color: 'white',
                      }}
                    >
                      {index + 1}
                    </div>
                  </Pin>
                </AdvancedMarker>
              );
            })}
          </Map>
        </APIProvider>
      </Paper>
    </div>
  );
}

export default MapDisplay;

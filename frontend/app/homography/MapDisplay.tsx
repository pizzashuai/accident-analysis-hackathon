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
import { Paper } from '@mantine/core';

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
  id: number;
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
  hoveredPairId?: number | null;
  pickingMode?: boolean;
  pendingPointA?: Point | null;
}

export function MapDisplay({
  width = '100%',
  height = 400,
  center = { lat: 1.3521, lng: 103.8198 }, // Default to Singapore
  zoom = 12,
  apiKey = 'AIzaSyCYrfgKpls8rM2nzl7qIFXhTF2jT9gbOgA',
  onMapClick,
  onMapHover,
  pairs = [],
  hoveredPairId = null,
  pickingMode = false,
  pendingPointA = null,
}: MapDisplayProps) {
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

  return (
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
      <APIProvider
        apiKey={apiKey}
        onLoad={() => console.log('Maps API has loaded.')}
      >
        <Map
          defaultZoom={zoom}
          defaultCenter={center}
          onCameraChanged={(ev: MapCameraChangedEvent) =>
            console.log(
              'camera changed:',
              ev.detail.center,
              'zoom:',
              ev.detail.zoom
            )
          }
          onClick={handleMapClick}
          onMousemove={handleMapMouseMove}
          mapId='homography-map'
          style={{ width: '100%', height: '100%' }}
          gestureHandling={pickingMode ? 'greedy' : 'auto'}
          disableDefaultUI={false}
        >
          {/* Render markers for paired points */}
          {pairs.map((pair, index) => (
            <AdvancedMarker
              key={pair.id}
              position={pair.b}
              title={`Point ${index + 1}`}
            >
              <Pin
                background={hoveredPairId === pair.id ? '#ff6b6b' : '#4dabf7'}
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
          ))}
        </Map>
      </APIProvider>
    </Paper>
  );
}

export default MapDisplay;

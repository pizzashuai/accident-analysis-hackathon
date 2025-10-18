import { useState, useRef, useEffect, useCallback } from 'react';
import {
  Stack,
  Group,
  Button,
  Text,
  Paper,
  Box,
  Table,
  ActionIcon,
  Card,
  Badge,
  Flex,
  CopyButton,
  Alert,
  TextInput,
} from '@mantine/core';
import { Dropzone, IMAGE_MIME_TYPE } from '@mantine/dropzone';
import {
  IconUpload,
  IconX,
  IconPhoto,
  IconTrash,
  IconCopy,
  IconCheck,
  IconDownload,
  IconSearch,
  IconMapPin,
} from '@tabler/icons-react';
import { MapDisplay } from './MapDisplay';

interface Point {
  xNorm: number;
  yNorm: number;
  x?: number; // rendered coordinates for display
  y?: number;
}

interface LatLngPoint {
  lat: number;
  lng: number;
}

interface PointPair {
  id: number;
  a: Point;
  b: LatLngPoint;
}

interface ImageData {
  url: string;
  file: File;
}

interface ImageMetrics {
  width: number;
  height: number;
  offsetX: number;
  offsetY: number;
}

export function HomographyPicker() {
  const [imageA, setImageA] = useState<ImageData | null>(null);
  const [imageLocation, setImageLocation] = useState<string>(
    '800 140th Ave NE, Bellevue, WA'
  );
  const [mapCenter, setMapCenter] = useState<LatLngPoint>({
    lat: 47.6169117,
    lng: -122.1432687,
  }); // Bellevue, WA
  const [mapZoom, setMapZoom] = useState(18);
  const [pickingMode, setPickingMode] = useState(false);
  const [pairs, setPairs] = useState<PointPair[]>([]);
  const [pendingPointA, setPendingPointA] = useState<Point | null>(null);
  const [hoveredPairId, setHoveredPairId] = useState<number | null>(null);
  const [currentCoordA, setCurrentCoordA] = useState<Point | null>(null);
  const [currentCoordB, setCurrentCoordB] = useState<LatLngPoint | null>(null);
  const [geocoding, setGeocoding] = useState(false);
  const [mapKey, setMapKey] = useState(0); // Key to force map re-render

  const imageRefA = useRef<HTMLImageElement>(null);
  const containerRefA = useRef<HTMLDivElement>(null);

  const [metricsA, setMetricsA] = useState<ImageMetrics>({
    width: 0,
    height: 0,
    offsetX: 0,
    offsetY: 0,
  });

  // Update metrics when images load or window resizes
  const updateMetrics = useCallback(() => {
    if (imageRefA.current && containerRefA.current) {
      const rect = imageRefA.current.getBoundingClientRect();
      setMetricsA({
        width: imageRefA.current.clientWidth,
        height: imageRefA.current.clientHeight,
        offsetX: rect.left,
        offsetY: rect.top,
      });
    }
  }, []);

  useEffect(() => {
    updateMetrics();
    window.addEventListener('resize', updateMetrics);
    return () => window.removeEventListener('resize', updateMetrics);
  }, [updateMetrics, imageA]);

  const handleDropA = (files: File[]) => {
    if (files[0]) {
      setImageA({ url: URL.createObjectURL(files[0]), file: files[0] });
      setPairs([]);
      setPendingPointA(null);
      setPickingMode(false);
    }
  };

  const normalizeCoordinates = (
    x: number,
    y: number,
    metrics: ImageMetrics
  ): Point => {
    const xNorm = Math.max(0, Math.min(1, x / metrics.width));
    const yNorm = Math.max(0, Math.min(1, y / metrics.height));
    return { x, y, xNorm, yNorm };
  };

  const denormalizeCoordinates = (point: Point, metrics: ImageMetrics) => {
    return {
      x: point.xNorm * metrics.width,
      y: point.yNorm * metrics.height,
    };
  };

  const handleImageClickA = (e: React.MouseEvent<HTMLDivElement>) => {
    const offsetX = e.nativeEvent.offsetX;
    const offsetY = e.nativeEvent.offsetY;

    console.log('[DEBUG] Image A clicked', {
      pickingMode,
      pendingPointA,
      clientX: e.clientX,
      clientY: e.clientY,
      offsetX,
      offsetY,
      metricsA,
    });

    if (!pickingMode || pendingPointA) {
      console.log(
        '[DEBUG] Ignoring click on A - pickingMode:',
        pickingMode,
        'pendingPointA exists:',
        !!pendingPointA
      );
      return;
    }

    const point = normalizeCoordinates(offsetX, offsetY, metricsA);
    console.log('[DEBUG] Point A set:', point);
    setPendingPointA(point);
    setCurrentCoordA(point);
  };

  const handleMapClick = (latLng: LatLngPoint) => {
    console.log('[DEBUG] Map clicked', {
      pickingMode,
      pendingPointA,
      lat: latLng.lat,
      lng: latLng.lng,
    });

    if (!pickingMode) {
      console.log('[DEBUG] Ignoring click on map - not in picking mode');
      return;
    }

    if (pendingPointA) {
      // Complete the pair
      const newPair: PointPair = {
        id: Date.now(),
        a: pendingPointA,
        b: latLng,
      };
      console.log('[DEBUG] Pair completed:', newPair);
      setPairs((prev) => [...prev, newPair]);
      setPendingPointA(null);
      setCurrentCoordA(null);
      setCurrentCoordB(null);
    } else {
      console.log('[DEBUG] No pending point A, just updating current coord B');
      setCurrentCoordB(latLng);
    }
  };

  const handleImageMoveA = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!pickingMode) return;
    const offsetX = e.nativeEvent.offsetX;
    const offsetY = e.nativeEvent.offsetY;
    const point = normalizeCoordinates(offsetX, offsetY, metricsA);
    setCurrentCoordA(point);
  };

  const handleMapHover = (latLng: LatLngPoint) => {
    if (!pickingMode) return;
    setCurrentCoordB(latLng);
  };

  const startNewPair = () => {
    console.log('[DEBUG] Starting new pair - resetting pendingPointA');
    setPendingPointA(null);
    setCurrentCoordA(null);
    setCurrentCoordB(null);
  };

  const deletePair = (id: number) => {
    setPairs((prev) => prev.filter((p) => p.id !== id));
  };

  const exportData = () => {
    const data = {
      pairs: pairs.map((pair, index) => ({
        id: index + 1,
        a: { xNorm: pair.a.xNorm, yNorm: pair.a.yNorm },
        b: { lat: pair.b.lat, lng: pair.b.lng },
      })),
      imagesMeta: {
        imageA: { name: imageA?.file.name, size: imageA?.file.size },
        metricsA,
      },
      mapMeta: {
        center: mapCenter,
        zoom: mapZoom,
      },
    };
    return JSON.stringify(data, null, 2);
  };

  const importData = (jsonString: string) => {
    try {
      const data = JSON.parse(jsonString);
      const importedPairs: PointPair[] = data.pairs.map((p: any) => ({
        id: Date.now() + p.id,
        a: { xNorm: p.a.xNorm, yNorm: p.a.yNorm },
        b: { lat: p.b.lat, lng: p.b.lng },
      }));
      setPairs(importedPairs);
      if (data.mapMeta) {
        setMapCenter(data.mapMeta.center);
        setMapZoom(data.mapMeta.zoom);
      }
    } catch (error) {
      console.error('Failed to import data:', error);
    }
  };

  const geocodeAddress = async () => {
    if (!imageLocation.trim()) return;

    setGeocoding(true);
    try {
      const response = await fetch(
        `https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(
          imageLocation
        )}&key=AIzaSyCYrfgKpls8rM2nzl7qIFXhTF2jT9gbOgA`
      );
      const data = await response.json();

      if (data.results && data.results.length > 0) {
        const location = data.results[0].geometry.location;
        setMapCenter({ lat: location.lat, lng: location.lng });
        setMapZoom(18);
        setMapKey((prev) => prev + 1); // Force map re-render with new center
      } else {
        console.error('No results found for address:', imageLocation);
        alert('Location not found. Please try a different address.');
      }
    } catch (error) {
      console.error('Geocoding error:', error);
      alert(
        'Failed to geocode address. Please check your internet connection.'
      );
    } finally {
      setGeocoding(false);
    }
  };

  const recenterMap = () => {
    setMapKey((prev) => prev + 1); // Force map to re-center to current mapCenter
  };

  const bothImagesLoaded = imageA;
  const hasMinimumPairs = pairs.length >= 4;

  return (
    <Stack gap='lg'>
      {/* Upload CCTV Image */}
      <Paper p='md' withBorder>
        <Stack gap='sm'>
          <Text fw={600} size='sm'>
            CCTV Image (A)
          </Text>
          {!imageA ? (
            <Dropzone
              onDrop={handleDropA}
              accept={IMAGE_MIME_TYPE}
              maxFiles={1}
            >
              <Group
                justify='center'
                gap='xs'
                mih={100}
                style={{ pointerEvents: 'none' }}
              >
                <Dropzone.Accept>
                  <IconUpload size={32} stroke={1.5} />
                </Dropzone.Accept>
                <Dropzone.Reject>
                  <IconX size={32} stroke={1.5} />
                </Dropzone.Reject>
                <Dropzone.Idle>
                  <IconPhoto size={32} stroke={1.5} />
                </Dropzone.Idle>
                <div>
                  <Text size='sm' inline>
                    Drag CCTV image here or click to select
                  </Text>
                </div>
              </Group>
            </Dropzone>
          ) : (
            <Box>
              <img
                src={imageA.url}
                alt='CCTV'
                style={{ maxWidth: '100%', height: 'auto' }}
              />
              <Button
                size='xs'
                variant='light'
                color='red'
                onClick={() => setImageA(null)}
                mt='xs'
                fullWidth
              >
                Remove
              </Button>
            </Box>
          )}
        </Stack>
      </Paper>

      {/* Image Location */}
      <Paper p='md' withBorder>
        <Stack gap='sm'>
          <Text fw={600} size='sm'>
            Image Location
          </Text>
          <Group gap='xs' align='flex-end'>
            <TextInput
              flex={1}
              placeholder='Enter address or location'
              value={imageLocation}
              onChange={(e) => setImageLocation(e.currentTarget.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  geocodeAddress();
                }
              }}
              leftSection={<IconMapPin size={16} />}
              description='Default: 800 140th Ave NE, Bellevue, WA'
            />
            <Button
              onClick={geocodeAddress}
              loading={geocoding}
              leftSection={<IconSearch size={16} />}
            >
              Search
            </Button>
            <Button
              onClick={recenterMap}
              variant='light'
              leftSection={<IconMapPin size={16} />}
            >
              Re-center
            </Button>
          </Group>
        </Stack>
      </Paper>

      {/* Pick Points Button */}
      <Group justify='center'>
        <Button
          disabled={!bothImagesLoaded}
          onClick={() => setPickingMode(!pickingMode)}
          color={pickingMode ? 'red' : 'blue'}
        >
          {pickingMode ? 'Stop Picking' : 'Pick Points'}
        </Button>
        {pickingMode && (
          <Button
            variant='light'
            onClick={startNewPair}
            disabled={!pendingPointA}
          >
            New Pair
          </Button>
        )}
      </Group>

      {/* Status Alert */}
      {pickingMode && (
        <Alert color={pendingPointA ? 'orange' : 'blue'}>
          {pendingPointA
            ? 'Now click a location on the Map (B) to complete the pair'
            : 'Click "New Pair" then click a point on CCTV (A) image'}
        </Alert>
      )}

      {/* Side-by-side Image and Map */}
      {bothImagesLoaded && (
        <Group grow align='flex-start'>
          <Card padding='md' withBorder>
            <Stack gap='xs'>
              <Text fw={600} size='sm'>
                CCTV Image (A)
              </Text>
              <Box
                ref={containerRefA}
                style={{
                  position: 'relative',
                  cursor:
                    pickingMode && !pendingPointA ? 'crosshair' : 'default',
                }}
                onClick={handleImageClickA}
                onMouseMove={handleImageMoveA}
              >
                <img
                  ref={imageRefA}
                  src={imageA.url}
                  alt='CCTV'
                  onLoad={updateMetrics}
                  style={{
                    maxWidth: '100%',
                    height: 'auto',
                    display: 'block',
                  }}
                />
                {/* Render markers */}
                {pairs.map((pair, index) => {
                  const pos = denormalizeCoordinates(pair.a, metricsA);
                  return (
                    <Box
                      key={pair.id}
                      style={{
                        position: 'absolute',
                        left: pos.x,
                        top: pos.y,
                        transform: 'translate(-50%, -50%)',
                        width: 24,
                        height: 24,
                        borderRadius: '50%',
                        backgroundColor:
                          hoveredPairId === pair.id ? '#ff6b6b' : '#4dabf7',
                        color: 'white',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '12px',
                        fontWeight: 'bold',
                        border: '2px solid white',
                        boxShadow: '0 2px 4px rgba(0,0,0,0.3)',
                        pointerEvents: 'none',
                      }}
                    >
                      {index + 1}
                    </Box>
                  );
                })}
                {/* Render pending point A marker */}
                {pendingPointA && (
                  <Box
                    style={{
                      position: 'absolute',
                      left: denormalizeCoordinates(pendingPointA, metricsA).x,
                      top: denormalizeCoordinates(pendingPointA, metricsA).y,
                      transform: 'translate(-50%, -50%)',
                      width: 24,
                      height: 24,
                      borderRadius: '50%',
                      backgroundColor: '#ffd43b',
                      color: '#000',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '12px',
                      fontWeight: 'bold',
                      border: '2px solid white',
                      boxShadow: '0 2px 4px rgba(0,0,0,0.3)',
                      pointerEvents: 'none',
                      animation: 'pulse 1s infinite',
                    }}
                  >
                    ?
                  </Box>
                )}
              </Box>
              {pickingMode && (
                <>
                  {pendingPointA && (
                    <Text size='sm' fw={600} c='yellow.7'>
                      ✓ Point A selected: ({pendingPointA.x?.toFixed(0)},{' '}
                      {pendingPointA.y?.toFixed(0)}) | Norm: (
                      {pendingPointA.xNorm.toFixed(4)},{' '}
                      {pendingPointA.yNorm.toFixed(4)})
                    </Text>
                  )}
                  {currentCoordA && (
                    <Text size='xs' c='dimmed'>
                      Hover: ({currentCoordA.x?.toFixed(0)},{' '}
                      {currentCoordA.y?.toFixed(0)}) | Norm: (
                      {currentCoordA.xNorm.toFixed(4)},{' '}
                      {currentCoordA.yNorm.toFixed(4)})
                    </Text>
                  )}
                </>
              )}
            </Stack>
          </Card>

          <Card padding='md' withBorder>
            <Stack gap='xs'>
              <Text fw={600} size='sm'>
                Google Map (B)
              </Text>
              <MapDisplay
                key={mapKey}
                height={500}
                center={mapCenter}
                zoom={mapZoom}
                onMapClick={handleMapClick}
                onMapHover={handleMapHover}
                pairs={pairs}
                hoveredPairId={hoveredPairId}
                pickingMode={pickingMode}
                pendingPointA={pendingPointA}
              />
              {currentCoordB && pickingMode && (
                <Text size='xs' c='dimmed'>
                  Hover: Lat: {currentCoordB.lat.toFixed(6)}, Lng:{' '}
                  {currentCoordB.lng.toFixed(6)}
                </Text>
              )}
            </Stack>
          </Card>
        </Group>
      )}

      {/* Pairs Table */}
      {pairs.length > 0 && (
        <Paper p='md' withBorder>
          <Stack gap='sm'>
            <Flex justify='space-between' align='center'>
              <Text fw={600}>Point Pairs ({pairs.length})</Text>
              <Badge color={hasMinimumPairs ? 'green' : 'red'}>
                {hasMinimumPairs ? 'Ready' : `Need ${4 - pairs.length} more`}
              </Badge>
            </Flex>
            <Table striped highlightOnHover>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>#</Table.Th>
                  <Table.Th>A (xNorm, yNorm)</Table.Th>
                  <Table.Th>B (Lat, Lng)</Table.Th>
                  <Table.Th>Actions</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {pairs.map((pair, index) => (
                  <Table.Tr
                    key={pair.id}
                    onMouseEnter={() => setHoveredPairId(pair.id)}
                    onMouseLeave={() => setHoveredPairId(null)}
                    style={{
                      backgroundColor:
                        hoveredPairId === pair.id
                          ? 'rgba(77, 171, 247, 0.1)'
                          : undefined,
                    }}
                  >
                    <Table.Td>{index + 1}</Table.Td>
                    <Table.Td>
                      ({pair.a.xNorm.toFixed(4)}, {pair.a.yNorm.toFixed(4)})
                    </Table.Td>
                    <Table.Td>
                      ({pair.b.lat.toFixed(6)}, {pair.b.lng.toFixed(6)})
                    </Table.Td>
                    <Table.Td>
                      <ActionIcon
                        color='red'
                        variant='subtle'
                        onClick={() => deletePair(pair.id)}
                      >
                        <IconTrash size={16} />
                      </ActionIcon>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Stack>
        </Paper>
      )}

      {/* Export Section */}
      {hasMinimumPairs && (
        <Paper p='md' withBorder>
          <Stack gap='sm'>
            <Text fw={600}>Review / Export</Text>
            <Text size='sm' c='dimmed'>
              {pairs.length} point pairs captured. Export data for homography
              transformation.
            </Text>
            <Group>
              <CopyButton value={exportData()}>
                {({ copied, copy }) => (
                  <Button
                    leftSection={
                      copied ? <IconCheck size={16} /> : <IconCopy size={16} />
                    }
                    color={copied ? 'teal' : 'blue'}
                    onClick={copy}
                  >
                    {copied ? 'Copied!' : 'Copy to Clipboard'}
                  </Button>
                )}
              </CopyButton>
              <Button
                leftSection={<IconDownload size={16} />}
                variant='light'
                onClick={() => {
                  const blob = new Blob([exportData()], {
                    type: 'application/json',
                  });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = 'homography-points.json';
                  a.click();
                }}
              >
                Download JSON
              </Button>
            </Group>
          </Stack>
        </Paper>
      )}
    </Stack>
  );
}

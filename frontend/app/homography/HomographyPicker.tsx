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
  Alert,
  TextInput,
  Loader,
} from '@mantine/core';
import { Dropzone, IMAGE_MIME_TYPE } from '@mantine/dropzone';
import {
  IconUpload,
  IconX,
  IconPhoto,
  IconTrash,
  IconMapPin,
  IconSearch,
  IconCamera,
  IconCheck,
} from '@tabler/icons-react';
import { MapDisplay } from './MapDisplay';
import { HomographyMatrixDisplay } from './HomographyMatrixDisplay';
import {
  useHomographySession,
  useCreateHomographySession,
  useUpdateHomographyPairs,
  useDeleteHomographyPair,
  useSolveHomography,
  useExtractFrame,
} from '~/hooks/useHomography';
import { useMediaPresignedUrl } from '~/hooks/useProjects';
import { useCustomToast } from '~/hooks/useCustomToast';
import type { HomographySessionPublic, HomographyPairPublic } from '~/client';

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
  id: string;
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

interface HomographyPickerProps {
  projectId: string;
  existingSession?: HomographySessionPublic;
}

export function HomographyPicker({
  projectId,
  existingSession,
}: HomographyPickerProps) {
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
  const [hoveredPairId, setHoveredPairId] = useState<string | null>(null);
  const [currentCoordA, setCurrentCoordA] = useState<Point | null>(null);
  const [currentCoordB, setCurrentCoordB] = useState<LatLngPoint | null>(null);
  const [geocoding, setGeocoding] = useState(false);
  const [mapKey, setMapKey] = useState(0); // Key to force map re-render
  const [saving, setSaving] = useState(false);
  const [screenshotUrl, setScreenshotUrl] = useState<string | null>(null);
  const [loadingScreenshot, setLoadingScreenshot] = useState(false);

  const imageRefA = useRef<HTMLImageElement>(null);
  const containerRefA = useRef<HTMLDivElement>(null);

  const [metricsA, setMetricsA] = useState<ImageMetrics>({
    width: 0,
    height: 0,
    offsetX: 0,
    offsetY: 0,
  });

  const { showToast } = useCustomToast();

  // Backend integration hooks
  const {
    data: session,
    isLoading: sessionLoading,
    refetch: refetchSession,
  } = useHomographySession(projectId);
  const createSession = useCreateHomographySession();
  const updatePairs = useUpdateHomographyPairs();
  const deletePair = useDeleteHomographyPair();
  const solveHomography = useSolveHomography();
  const extractFrame = useExtractFrame();
  const getMediaPresignedUrl = useMediaPresignedUrl();

  // Fetch screenshot URL from media asset
  const fetchScreenshotUrl = async (screenshotAssetId: string) => {
    setLoadingScreenshot(true);
    try {
      const response = await getMediaPresignedUrl.mutateAsync({
        projectId,
        mediaAssetId: screenshotAssetId,
      });
      setScreenshotUrl((response as { url: string }).url);
      // Trigger metrics update after a short delay to ensure image is loaded
      setTimeout(() => {
        updateMetrics();
      }, 100);
    } catch (error) {
      console.error('Failed to fetch screenshot URL:', error);
      showToast('Failed to load screenshot', 'error');
    } finally {
      setLoadingScreenshot(false);
    }
  };

  // Load existing session data on mount
  useEffect(() => {
    if (session) {
      // Convert backend pairs to frontend format
      const convertedPairs: PointPair[] = session.pairs.map(
        (pair: HomographyPairPublic) => ({
          id: pair.id,
          a: { xNorm: pair.image_x_norm, yNorm: pair.image_y_norm },
          b: { lat: pair.map_lat, lng: pair.map_lng },
        })
      );
      setPairs(convertedPairs);

      // Load screenshot if available
      if (session.screenshot_asset_id) {
        fetchScreenshotUrl(session.screenshot_asset_id);
      }
    }
  }, [session]);

  // Update metrics when images load or window resizes
  const updateMetrics = useCallback(() => {
    if (imageRefA.current && containerRefA.current) {
      const rect = imageRefA.current.getBoundingClientRect();
      const newMetrics = {
        width: imageRefA.current.clientWidth,
        height: imageRefA.current.clientHeight,
        offsetX: rect.left,
        offsetY: rect.top,
      };

      // Only update if metrics have actually changed to avoid unnecessary re-renders
      setMetricsA((prevMetrics) => {
        if (
          prevMetrics.width !== newMetrics.width ||
          prevMetrics.height !== newMetrics.height ||
          prevMetrics.offsetX !== newMetrics.offsetX ||
          prevMetrics.offsetY !== newMetrics.offsetY
        ) {
          return newMetrics;
        }
        return prevMetrics;
      });
    }
  }, []);

  useEffect(() => {
    updateMetrics();
    window.addEventListener('resize', updateMetrics);
    return () => window.removeEventListener('resize', updateMetrics);
  }, [updateMetrics, imageA]);

  // Update metrics when screenshot URL changes
  useEffect(() => {
    if (screenshotUrl) {
      // Multiple attempts to ensure image is loaded and metrics are calculated
      const timers = [
        setTimeout(() => updateMetrics(), 100),
        setTimeout(() => updateMetrics(), 500),
        setTimeout(() => updateMetrics(), 1000),
      ];

      return () => {
        timers.forEach((timer) => clearTimeout(timer));
      };
    }
  }, [screenshotUrl, updateMetrics]);

  // Additional effect to update metrics when pairs are loaded
  useEffect(() => {
    if (pairs.length > 0 && (imageA || screenshotUrl)) {
      // Ensure metrics are calculated when pairs exist
      const timer = setTimeout(() => {
        updateMetrics();
      }, 200);
      return () => clearTimeout(timer);
    }
  }, [pairs.length, imageA, screenshotUrl, updateMetrics]);

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

    if (!pickingMode || pendingPointA) {
      return;
    }

    const point = normalizeCoordinates(offsetX, offsetY, metricsA);
    setPendingPointA(point);
    setCurrentCoordA(point);
  };

  const handleMapClick = (latLng: LatLngPoint) => {
    if (!pickingMode) {
      return;
    }

    if (pendingPointA) {
      // Complete the pair
      const newPair: PointPair = {
        id: Date.now().toString(),
        a: pendingPointA,
        b: latLng,
      };
      const updatedPairs = [...pairs, newPair];
      setPairs(updatedPairs);
      setPendingPointA(null);
      setCurrentCoordA(null);
      setCurrentCoordB(null);

      // Auto-save pairs to backend
      savePairsToBackend(updatedPairs);
    } else {
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
    setPendingPointA(null);
    setCurrentCoordA(null);
    setCurrentCoordB(null);
  };

  const deletePairLocal = async (id: string) => {
    const updatedPairs = pairs.filter((p) => p.id !== id);
    setPairs(updatedPairs);

    // Delete from backend
    try {
      await deletePair.mutateAsync(id);
      showToast('Pair deleted successfully', 'success');
    } catch (error) {
      showToast('Failed to delete pair', 'error');
      // Revert local change
      setPairs(pairs);
    }
  };

  const savePairsToBackend = async (pairsToSave: PointPair[]) => {
    if (!session) return;

    setSaving(true);
    try {
      const pairsData = pairsToSave.map((pair, index) => ({
        image_x_norm: pair.a.xNorm,
        image_y_norm: pair.a.yNorm,
        map_lat: pair.b.lat,
        map_lng: pair.b.lng,
        order_idx: index,
      }));

      await updatePairs.mutateAsync({
        sessionId: session.id,
        pairsData,
      });

      showToast('Pairs saved successfully', 'success');
    } catch (error) {
      showToast('Failed to save pairs', 'error');
    } finally {
      setSaving(false);
    }
  };

  const handleSolveHomography = async () => {
    if (!session) return;

    try {
      const result = await solveHomography.mutateAsync(session.id);
      if (result.success) {
        showToast('Homography solved successfully!', 'success');
        refetchSession();
      } else {
        showToast(
          result.error_message || 'Failed to solve homography',
          'error'
        );
      }
    } catch (error) {
      showToast('Failed to solve homography', 'error');
    }
  };

  const handleExtractFrame = async () => {
    try {
      await extractFrame.mutateAsync(projectId);
      showToast('Frame extracted successfully', 'success');
      refetchSession();
    } catch (error) {
      showToast('Failed to extract frame', 'error');
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
        setMapKey((prev) => prev + 1);
      } else {
        showToast(
          'Location not found. Please try a different address.',
          'error'
        );
      }
    } catch (error) {
      showToast(
        'Failed to geocode address. Please check your internet connection.',
        'error'
      );
    } finally {
      setGeocoding(false);
    }
  };

  const recenterMap = () => {
    setMapKey((prev) => prev + 1);
  };

  const bothImagesLoaded = imageA || screenshotUrl;
  const hasMinimumPairs = pairs.length >= 4;
  const isSolved = session?.status === 'solved';

  if (sessionLoading) {
    return (
      <Stack align='center' gap='md' py='xl'>
        <Loader size='lg' />
        <Text c='dimmed'>Loading homography session...</Text>
      </Stack>
    );
  }

  return (
    <Stack gap='lg'>
      {/* Session Status */}
      <Paper p='md' withBorder>
        <Group justify='space-between' align='center'>
          <Group gap='xs'>
            <Text fw={600} size='md'>
              Homography Configuration
            </Text>
            <Badge
              color={
                isSolved
                  ? 'green'
                  : session?.status === 'draft'
                    ? 'yellow'
                    : 'gray'
              }
              variant='light'
            >
              {isSolved
                ? 'Solved'
                : session?.status === 'draft'
                  ? 'Draft'
                  : 'Not configured'}
            </Badge>
          </Group>

          {!session && (
            <Button
              onClick={() => createSession.mutate(projectId)}
              loading={createSession.isPending}
              leftSection={<IconCheck size={16} />}
            >
              Initialize Session
            </Button>
          )}
        </Group>
      </Paper>

      {/* Extract Frame Section */}
      {session && !imageA && !screenshotUrl && (
        <Paper p='md' withBorder>
          <Stack gap='sm'>
            <Text fw={600} size='sm'>
              Screenshot
            </Text>
            <Group gap='xs'>
              <Button
                onClick={handleExtractFrame}
                loading={extractFrame.isPending}
                leftSection={<IconCamera size={16} />}
                variant='light'
              >
                Extract Video Frame
              </Button>
              <Text size='sm' c='dimmed'>
                Extract the first frame from your uploaded video, or upload a
                screenshot manually below.
              </Text>
            </Group>
          </Stack>
        </Paper>
      )}

      {/* Upload CCTV Image */}
      <Paper p='md' withBorder>
        <Stack gap='sm'>
          <Text fw={600} size='sm'>
            CCTV Image (A)
          </Text>
          {imageA ? (
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
          ) : screenshotUrl ? (
            <Box>
              {loadingScreenshot ? (
                <Group justify='center' p='xl'>
                  <Loader size='md' />
                  <Text size='sm' c='dimmed'>
                    Loading screenshot...
                  </Text>
                </Group>
              ) : (
                <img
                  src={screenshotUrl}
                  alt='Video Screenshot'
                  style={{ maxWidth: '100%', height: 'auto' }}
                  onError={() => {
                    showToast('Screenshot failed to load', 'error');
                  }}
                />
              )}
              <Text size='xs' c='dimmed' mt='xs'>
                First frame extracted from video
              </Text>
            </Box>
          ) : (
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

      {/* Save Status */}
      {saving && <Alert color='blue'>Saving pairs to backend...</Alert>}

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
                  src={imageA?.url || screenshotUrl || ''}
                  alt='CCTV'
                  onLoad={() => {
                    // Multiple attempts to ensure metrics are calculated after image loads
                    updateMetrics();
                    setTimeout(() => updateMetrics(), 100);
                    setTimeout(() => updateMetrics(), 300);
                  }}
                  style={{
                    maxWidth: '100%',
                    height: 'auto',
                    display: 'block',
                  }}
                />
                {/* Render markers */}
                {pairs.map((pair, index) => {
                  // Use fallback metrics if not properly initialized
                  const effectiveMetrics =
                    metricsA && metricsA.width > 0 && metricsA.height > 0
                      ? metricsA
                      : { width: 400, height: 300, offsetX: 0, offsetY: 0 }; // Fallback dimensions

                  const pos = denormalizeCoordinates(pair.a, effectiveMetrics);
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
                      left: denormalizeCoordinates(
                        pendingPointA,
                        metricsA && metricsA.width > 0 && metricsA.height > 0
                          ? metricsA
                          : { width: 400, height: 300, offsetX: 0, offsetY: 0 }
                      ).x,
                      top: denormalizeCoordinates(
                        pendingPointA,
                        metricsA && metricsA.width > 0 && metricsA.height > 0
                          ? metricsA
                          : { width: 400, height: 300, offsetX: 0, offsetY: 0 }
                      ).y,
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
                      {pendingPointA.y?.toFixed(0)}) | Normalized: (
                      {pendingPointA.xNorm.toFixed(4)},{' '}
                      {pendingPointA.yNorm.toFixed(4)})
                    </Text>
                  )}
                  {currentCoordA && (
                    <Text size='xs' c='dimmed'>
                      Hover: ({currentCoordA.x?.toFixed(0)},{' '}
                      {currentCoordA.y?.toFixed(0)}) | Normalized: (
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
            <Text size='xs' c='dimmed'>
              Normalized coordinates are values between 0 and 1 representing the
              position within the image (0,0 = top-left, 1,1 = bottom-right)
            </Text>
            <Table striped highlightOnHover>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>#</Table.Th>
                  <Table.Th>A (Normalized Coordinates)</Table.Th>
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
                        onClick={() => deletePairLocal(pair.id)}
                        loading={deletePair.isPending}
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

      {/* Solve Homography Section */}
      {hasMinimumPairs && !isSolved && (
        <Paper p='md' withBorder>
          <Stack gap='sm'>
            <Text fw={600}>Solve Homography</Text>
            <Text size='sm' c='dimmed'>
              {pairs.length} point pairs captured. Solve the homography
              transformation.
            </Text>
            <Button
              onClick={handleSolveHomography}
              loading={solveHomography.isPending}
              leftSection={<IconCheck size={16} />}
              color='green'
            >
              Solve Homography
            </Button>
          </Stack>
        </Paper>
      )}

      {/* Show solved matrix */}
      {isSolved && session?.model && (
        <HomographyMatrixDisplay
          matrix={session.model.matrix_data}
          error={session.model.reprojection_error}
          inlierCount={session.model.meta?.inlier_count}
          totalPairs={session.model.meta?.total_pairs}
        />
      )}
    </Stack>
  );
}

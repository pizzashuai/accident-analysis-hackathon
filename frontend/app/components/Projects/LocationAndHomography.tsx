import { useState, useRef, useEffect, useCallback } from 'react';
import {
  Button,
  Stack,
  Group,
  Text,
  NumberInput,
  Paper,
  TextInput,
  Box,
  Table,
  ActionIcon,
  Card,
  Badge,
  Flex,
  Alert,
  Loader,
} from '@mantine/core';
import { Dropzone, IMAGE_MIME_TYPE } from '@mantine/dropzone';
import { useForm } from '@mantine/form';
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
import {
  APIProvider,
  Map,
  AdvancedMarker,
  Pin,
} from '@vis.gl/react-google-maps';
import type { MapMouseEvent } from '@vis.gl/react-google-maps';
import { MapDisplay } from '~/homography/MapDisplay';
import {
  useSetProjectLocation,
  useMediaPresignedUrl,
} from '~/hooks/useProjects';
import {
  useHomographySession,
  useCreateHomographySession,
  useUpdateHomographyPairs,
  useDeleteHomographyPair,
  useSolveHomography,
  useExtractFrame,
} from '~/hooks/useHomography';
import { useCustomToast } from '~/hooks/useCustomToast';
import type { HomographySessionPublic, HomographyPairPublic } from '~/client';

interface LocationAndHomographyProps {
  projectId: string;
  project: any;
  onLocationSet?: () => void;
}

interface Point {
  xNorm: number;
  yNorm: number;
  x?: number;
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

export function LocationAndHomography({
  projectId,
  project,
  onLocationSet,
}: LocationAndHomographyProps) {
  // Location state
  const [isSubmittingLocation, setIsSubmittingLocation] = useState(false);
  const [mapCenter, setMapCenter] = useState<{ lat: number; lng: number }>({
    lat: project.location?.lat || 1.3521,
    lng: project.location?.lon || 103.8198,
  });
  const [markerPosition, setMarkerPosition] = useState<{
    lat: number;
    lng: number;
  } | null>(
    project.location?.lat && project.location?.lon
      ? { lat: project.location.lat, lng: project.location.lon }
      : null
  );
  const [mapZoom, setMapZoom] = useState(18);
  const [autocomplete, setAutocomplete] =
    useState<google.maps.places.Autocomplete | null>(null);

  // Homography state
  const [imageA, setImageA] = useState<ImageData | null>(null);
  const [pickingMode, setPickingMode] = useState(false);
  const [pairs, setPairs] = useState<PointPair[]>([]);
  const [pendingPointA, setPendingPointA] = useState<Point | null>(null);
  const [hoveredPairId, setHoveredPairId] = useState<string | null>(null);
  const [currentCoordA, setCurrentCoordA] = useState<Point | null>(null);
  const [currentCoordB, setCurrentCoordB] = useState<LatLngPoint | null>(null);
  const [saving, setSaving] = useState(false);
  const [screenshotUrl, setScreenshotUrl] = useState<string | null>(null);
  const [loadingScreenshot, setLoadingScreenshot] = useState(false);

  const imageRefA = useRef<HTMLImageElement>(null);
  const containerRefA = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const [metricsA, setMetricsA] = useState<ImageMetrics>({
    width: 0,
    height: 0,
    offsetX: 0,
    offsetY: 0,
  });

  const { showToast } = useCustomToast();
  const apiKey = import.meta.env.VITE_GOOGLE_MAP_KEY;

  // Location form
  const locationForm = useForm({
    initialValues: {
      addr_line: project.location?.addr_line || '',
      lat: project.location?.lat || undefined,
      lon: project.location?.lon || undefined,
    },
    validate: {
      lat: (value) => {
        if (value !== undefined && (value < -90 || value > 90)) {
          return 'Latitude must be between -90 and 90';
        }
        return null;
      },
      lon: (value) => {
        if (value !== undefined && (value < -180 || value > 180)) {
          return 'Longitude must be between -180 and 180';
        }
        return null;
      },
    },
  });

  // Backend integration hooks
  const setLocation = useSetProjectLocation();
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

  // Initialize autocomplete when Google Maps API is available
  useEffect(() => {
    const initAutocomplete = () => {
      if (
        window.google &&
        window.google.maps &&
        window.google.maps.places &&
        inputRef.current &&
        !autocomplete
      ) {
        const autocompleteInstance = new google.maps.places.Autocomplete(
          inputRef.current,
          {
            types: ['geocode', 'establishment'],
          }
        );

        autocompleteInstance.addListener('place_changed', () => {
          const place = autocompleteInstance.getPlace();
          handlePlaceSelected(place);
        });

        setAutocomplete(autocompleteInstance);
      }
    };

    initAutocomplete();

    if (!window.google?.maps?.places) {
      const timer = setTimeout(initAutocomplete, 1000);
      return () => clearTimeout(timer);
    }
  }, [autocomplete]);

  // Update form and map when coordinates change manually
  useEffect(() => {
    if (
      locationForm.values.lat !== undefined &&
      locationForm.values.lon !== undefined
    ) {
      const newPos = {
        lat: locationForm.values.lat,
        lng: locationForm.values.lon,
      };
      setMarkerPosition(newPos);
      setMapCenter(newPos);
    }
  }, [locationForm.values.lat, locationForm.values.lon]);

  // Fetch screenshot URL from media asset
  const fetchScreenshotUrl = async (screenshotAssetId: string) => {
    setLoadingScreenshot(true);
    try {
      const response = await getMediaPresignedUrl.mutateAsync({
        projectId,
        mediaAssetId: screenshotAssetId,
      });
      setScreenshotUrl((response as { url: string }).url);
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
      const convertedPairs: PointPair[] = (session.pairs || []).map(
        (pair: HomographyPairPublic) => ({
          id: pair.id.toString(),
          a: { xNorm: pair.image_x_norm, yNorm: pair.image_y_norm },
          b: { lat: pair.map_lat, lng: pair.map_lng },
        })
      );
      setPairs(convertedPairs);

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

  useEffect(() => {
    if (screenshotUrl) {
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

  useEffect(() => {
    if (pairs.length > 0 && (imageA || screenshotUrl)) {
      const timer = setTimeout(() => {
        updateMetrics();
      }, 200);
      return () => clearTimeout(timer);
    }
  }, [pairs.length, imageA, screenshotUrl, updateMetrics]);

  const handlePlaceSelected = (
    place: google.maps.places.PlaceResult | null
  ) => {
    if (place?.geometry?.location) {
      const lat = place.geometry.location.lat();
      const lng = place.geometry.location.lng();
      const address = place.formatted_address || '';

      locationForm.setFieldValue('addr_line', address);
      locationForm.setFieldValue('lat', lat);
      locationForm.setFieldValue('lon', lng);

      setMapCenter({ lat, lng });
      setMarkerPosition({ lat, lng });
    }
  };

  const handleMapClick = (event: MapMouseEvent) => {
    if (event.detail.latLng) {
      const lat = event.detail.latLng.lat;
      const lng = event.detail.latLng.lng;

      locationForm.setFieldValue('lat', lat);
      locationForm.setFieldValue('lon', lng);

      setMarkerPosition({ lat, lng });
    }
  };

  const handleLocationSubmit = async (values: typeof locationForm.values) => {
    setIsSubmittingLocation(true);
    try {
      await setLocation.mutateAsync({
        projectId,
        locationData: {
          addr_line: values.addr_line || undefined,
          lat: values.lat || undefined,
          lon: values.lon || undefined,
          source: 'user',
        },
      });
      showToast('Location updated successfully', 'success');
      onLocationSet?.();
    } catch (error) {
      showToast('Failed to update location', 'error');
    } finally {
      setIsSubmittingLocation(false);
    }
  };

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

  const handleHomographyMapClick = (latLng: LatLngPoint) => {
    if (!pickingMode) {
      return;
    }

    if (pendingPointA) {
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

    try {
      await deletePair.mutateAsync(id);
      showToast('Pair deleted successfully', 'success');
    } catch (error) {
      showToast('Failed to delete pair', 'error');
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

  // Show empty state when no homography session exists
  if (!session) {
    return (
      <Stack gap='lg'>
        {/* Location Section */}
        <Paper p='sm' withBorder>
          <Stack gap='md'>
            <div>
              <Text size='lg' fw={600} mb='xs'>
                Set Location
              </Text>
            </div>

            {project.location?.lat && project.location?.lon && (
              <Alert color='green' icon={<IconCheck size={16} />}>
                Location set:{' '}
                {project.location.addr_line ||
                  `${project.location.lat.toFixed(6)}, ${project.location.lon.toFixed(6)}`}
              </Alert>
            )}

            <APIProvider apiKey={apiKey} libraries={['places']}>
              <form onSubmit={locationForm.onSubmit(handleLocationSubmit)}>
                <Stack gap='md'>
                  <TextInput
                    ref={inputRef}
                    label='Address'
                    placeholder='Search for an address or place...'
                    {...locationForm.getInputProps('addr_line')}
                  />

                  {/* Interactive Map */}
                  <div>
                    <Text size='sm' fw={500} mb={4}>
                      Map
                    </Text>
                    <Paper
                      withBorder
                      style={{
                        width: '100%',
                        height: 300,
                        position: 'relative',
                        overflow: 'hidden',
                      }}
                    >
                      <Map
                        defaultZoom={mapZoom}
                        center={mapCenter}
                        onClick={handleMapClick}
                        mapId='location-picker-map'
                        style={{ width: '100%', height: '100%' }}
                        gestureHandling='greedy'
                        disableDefaultUI={false}
                      >
                        {markerPosition && (
                          <AdvancedMarker
                            position={markerPosition}
                            title='Selected Location'
                          >
                            <Pin
                              background='#228be6'
                              borderColor='#ffffff'
                              glyphColor='#ffffff'
                            />
                          </AdvancedMarker>
                        )}
                      </Map>
                    </Paper>
                  </div>

                  <Group grow>
                    <NumberInput
                      label='Latitude'
                      placeholder='e.g., 40.7128'
                      decimalScale={6}
                      min={-90}
                      max={90}
                      {...locationForm.getInputProps('lat')}
                    />
                    <NumberInput
                      label='Longitude'
                      placeholder='e.g., -74.0060'
                      decimalScale={6}
                      min={-180}
                      max={180}
                      {...locationForm.getInputProps('lon')}
                    />
                  </Group>

                  <Button type='submit' loading={isSubmittingLocation}>
                    {project.location ? 'Update Location' : 'Set Location'}
                  </Button>
                </Stack>
              </form>
            </APIProvider>
          </Stack>
        </Paper>

        {/* Empty State for Homography Configuration */}
        <Paper p='md' withBorder>
          <Stack gap='md' align='center'>
            <div style={{ textAlign: 'center' }}>
              <IconMapPin size={48} color='var(--mantine-color-gray-5)' />
              <Text size='lg' fw={600} mt='md' mb='xs'>
                Configure Homography Mapping
              </Text>
              <Text size='sm' c='dimmed' mb='lg'>
                Set up point mapping between your video frame and real-world
                coordinates to enable accurate speed calculations.
              </Text>
            </div>

            <Alert
              color='blue'
              icon={<IconMapPin size={16} />}
              style={{ width: '100%' }}
            >
              <Text size='sm'>
                <strong>Next steps:</strong>
                <br />
                1. Set the location above (if not already done)
                <br />
                2. Upload a video and capture a key frame
                <br />
                3. Initialize homography session to start mapping points
              </Text>
            </Alert>

            <Group gap='sm'>
              <Button
                onClick={() => createSession.mutate(projectId)}
                loading={createSession.isPending}
                leftSection={<IconCheck size={16} />}
                disabled={!project.location}
              >
                Initialize Homography Session
              </Button>
            </Group>

            {!project.location && (
              <Alert color='yellow' icon={<IconMapPin size={16} />}>
                Please set the location first before initializing homography
                session.
              </Alert>
            )}
          </Stack>
        </Paper>
      </Stack>
    );
  }

  return (
    <Stack gap='lg'>
      {/* Location Section */}
      <Paper p='sm' withBorder>
        <Stack gap='md'>
          <div>
            <Text size='lg' fw={600} mb='xs'>
              Set Location
            </Text>
          </div>

          {project.location?.lat && project.location?.lon && (
            <Alert color='green' icon={<IconCheck size={16} />}>
              Location set:{' '}
              {project.location.addr_line ||
                `${project.location.lat.toFixed(6)}, ${project.location.lon.toFixed(6)}`}
            </Alert>
          )}

          <APIProvider apiKey={apiKey} libraries={['places']}>
            <form onSubmit={locationForm.onSubmit(handleLocationSubmit)}>
              <Stack gap='md'>
                <TextInput
                  ref={inputRef}
                  label='Address'
                  placeholder='Search for an address or place...'
                  {...locationForm.getInputProps('addr_line')}
                />

                {/* Interactive Map */}
                <div>
                  <Text size='sm' fw={500} mb={4}>
                    Map
                  </Text>
                  <Paper
                    withBorder
                    style={{
                      width: '100%',
                      height: 300,
                      position: 'relative',
                      overflow: 'hidden',
                    }}
                  >
                    <Map
                      defaultZoom={mapZoom}
                      center={mapCenter}
                      onClick={handleMapClick}
                      mapId='location-picker-map'
                      style={{ width: '100%', height: '100%' }}
                      gestureHandling='greedy'
                      disableDefaultUI={false}
                    >
                      {markerPosition && (
                        <AdvancedMarker
                          position={markerPosition}
                          title='Selected Location'
                        >
                          <Pin
                            background='#228be6'
                            borderColor='#ffffff'
                            glyphColor='#ffffff'
                          />
                        </AdvancedMarker>
                      )}
                    </Map>
                  </Paper>
                </div>

                <Group grow>
                  <NumberInput
                    label='Latitude'
                    placeholder='e.g., 40.7128'
                    decimalScale={6}
                    min={-90}
                    max={90}
                    {...locationForm.getInputProps('lat')}
                  />
                  <NumberInput
                    label='Longitude'
                    placeholder='e.g., -74.0060'
                    decimalScale={6}
                    min={-180}
                    max={180}
                    {...locationForm.getInputProps('lon')}
                  />
                </Group>

                <Button type='submit' loading={isSubmittingLocation}>
                  {project.location ? 'Update Location' : 'Set Location'}
                </Button>
              </Stack>
            </form>
          </APIProvider>
        </Stack>
      </Paper>

      {/* Homography Configuration & Matrix Section */}
      <Paper p='sm' withBorder>
        <Stack gap='md'>
          <div>
            <Text size='lg' fw={600} mb='xs'>
              Homography Configuration
            </Text>
            <Text size='sm' c='dimmed'>
              Map points from your video frame to real-world coordinates.
            </Text>
          </div>

          {/* Session Status */}
          <Group justify='space-between' align='center'>
            <Group gap='xs'>
              <Text fw={600} size='sm'>
                Status
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
          </Group>

          {(session as any)?.status === 'solved' && (
            <Alert color='green' icon={<IconCheck size={16} />}>
              Homography solved successfully! You can now proceed to run
              processing.
            </Alert>
          )}

          {!project.media_assets?.some(
            (asset: any) => asset.kind === 'image'
          ) && (
            <Alert color='red' icon={<IconCheck size={16} />}>
              Please capture a key frame first (Step 2).
            </Alert>
          )}

          {!project.location && (
            <Alert color='red' icon={<IconCheck size={16} />}>
              Please set the location first above.
            </Alert>
          )}
        </Stack>
      </Paper>

      {/* Extract Frame Section */}
      {session && !imageA && !screenshotUrl && (
        <Paper p='sm' withBorder>
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
      {!imageA && !screenshotUrl && (
        <Paper p='sm' withBorder>
          <Stack gap='sm'>
            <Text fw={600} size='sm'>
              CCTV Image
            </Text>
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
          </Stack>
        </Paper>
      )}

      {/* Pick Points Button */}
      <Group justify='center'>
        <Button
          disabled={!bothImagesLoaded || !project.location}
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
      {bothImagesLoaded && project.location && (
        <Group grow align='flex-start'>
          <Card padding='md' withBorder style={{ flex: '2' }}>
            <Stack gap='xs'>
              <Group justify='space-between' align='center'>
                <Text fw={600} size='sm'>
                  CCTV Image (A)
                </Text>
                {imageA && (
                  <Button
                    size='xs'
                    variant='light'
                    color='red'
                    onClick={() => setImageA(null)}
                    leftSection={<IconTrash size={12} />}
                  >
                    Remove
                  </Button>
                )}
              </Group>
              {loadingScreenshot ? (
                <Group justify='center' p='xl'>
                  <Loader size='md' />
                  <Text size='sm' c='dimmed'>
                    Loading screenshot...
                  </Text>
                </Group>
              ) : (
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
                      updateMetrics();
                      setTimeout(() => updateMetrics(), 100);
                      setTimeout(() => updateMetrics(), 300);
                    }}
                    onError={() => {
                      if (screenshotUrl) {
                        showToast('Screenshot failed to load', 'error');
                      }
                    }}
                    style={{
                      width: '100%',
                      height: 'auto',
                      display: 'block',
                      maxHeight: '600px',
                      objectFit: 'contain',
                    }}
                  />
                  {/* Render markers */}
                  {pairs.map((pair, index) => {
                    const effectiveMetrics =
                      metricsA && metricsA.width > 0 && metricsA.height > 0
                        ? metricsA
                        : { width: 400, height: 300, offsetX: 0, offsetY: 0 };

                    const pos = denormalizeCoordinates(
                      pair.a,
                      effectiveMetrics
                    );
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
                            : {
                                width: 400,
                                height: 300,
                                offsetX: 0,
                                offsetY: 0,
                              }
                        ).x,
                        top: denormalizeCoordinates(
                          pendingPointA,
                          metricsA && metricsA.width > 0 && metricsA.height > 0
                            ? metricsA
                            : {
                                width: 400,
                                height: 300,
                                offsetX: 0,
                                offsetY: 0,
                              }
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
              )}
              {screenshotUrl && !imageA && (
                <Text size='xs' c='dimmed'>
                  First frame extracted from video
                </Text>
              )}
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

          <Card padding='md' withBorder style={{ flex: '1' }}>
            <Stack gap='xs'>
              <Text fw={600} size='sm'>
                Google Map (B) - Using Location Above
              </Text>
              <MapDisplay
                height={400}
                center={mapCenter}
                zoom={mapZoom}
                onMapClick={handleHomographyMapClick}
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
        <Paper p='sm' withBorder>
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
        <Paper p='sm' withBorder>
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
    </Stack>
  );
}

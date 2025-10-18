import { useState, useRef, useEffect } from 'react';
import {
  Button,
  Stack,
  Group,
  Text,
  NumberInput,
  Paper,
  TextInput,
} from '@mantine/core';
import { useForm } from '@mantine/form';
import { useSetProjectLocation } from '~/hooks/useProjects';
import { useCustomToast } from '~/hooks/useCustomToast';
import {
  APIProvider,
  Map,
  AdvancedMarker,
  Pin,
} from '@vis.gl/react-google-maps';
import type { MapMouseEvent } from '@vis.gl/react-google-maps';

interface LocationPickerProps {
  projectId: string;
  initialLocation?: {
    addr_line?: string;
    lat?: number;
    lon?: number;
  } | null;
  onLocationSet?: () => void;
}

export function LocationPicker({
  projectId,
  initialLocation,
  onLocationSet,
}: LocationPickerProps) {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [mapCenter, setMapCenter] = useState<{ lat: number; lng: number }>({
    lat: initialLocation?.lat || 1.3521,
    lng: initialLocation?.lon || 103.8198,
  });
  const [markerPosition, setMarkerPosition] = useState<{
    lat: number;
    lng: number;
  } | null>(
    initialLocation?.lat && initialLocation?.lon
      ? { lat: initialLocation.lat, lng: initialLocation.lon }
      : null
  );
  const [autocomplete, setAutocomplete] =
    useState<google.maps.places.Autocomplete | null>(null);
  const setLocation = useSetProjectLocation();
  const { showToast } = useCustomToast();
  const mapRef = useRef<google.maps.Map | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const apiKey = import.meta.env.VITE_GOOGLE_MAP_KEY;

  const form = useForm({
    initialValues: {
      addr_line: initialLocation?.addr_line || '',
      lat: initialLocation?.lat || undefined,
      lon: initialLocation?.lon || undefined,
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

    // Try to initialize immediately
    initAutocomplete();

    // If not ready, try again after a short delay
    if (!window.google?.maps?.places) {
      const timer = setTimeout(initAutocomplete, 1000);
      return () => clearTimeout(timer);
    }
  }, [autocomplete]);

  // Update form and map when coordinates change manually
  useEffect(() => {
    if (form.values.lat !== undefined && form.values.lon !== undefined) {
      const newPos = { lat: form.values.lat, lng: form.values.lon };
      setMarkerPosition(newPos);
      setMapCenter(newPos);
    }
  }, [form.values.lat, form.values.lon]);

  const handlePlaceSelected = (
    place: google.maps.places.PlaceResult | null
  ) => {
    if (place?.geometry?.location) {
      const lat = place.geometry.location.lat();
      const lng = place.geometry.location.lng();
      const address = place.formatted_address || '';

      // Update form values
      form.setFieldValue('addr_line', address);
      form.setFieldValue('lat', lat);
      form.setFieldValue('lon', lng);

      // Update map
      setMapCenter({ lat, lng });
      setMarkerPosition({ lat, lng });
    }
  };

  const handleMapClick = (event: MapMouseEvent) => {
    if (event.detail.latLng) {
      const lat = event.detail.latLng.lat;
      const lng = event.detail.latLng.lng;

      // Update form values
      form.setFieldValue('lat', lat);
      form.setFieldValue('lon', lng);

      // Update marker position
      setMarkerPosition({ lat, lng });
    }
  };

  const handleSubmit = async (values: typeof form.values) => {
    setIsSubmitting(true);
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
      setIsSubmitting(false);
    }
  };

  return (
    <Stack gap='md'>
      <Text size='sm' fw={500}>
        Set Project Location
      </Text>

      <APIProvider apiKey={apiKey} libraries={['places']}>
        <form onSubmit={form.onSubmit(handleSubmit)}>
          <Stack gap='md'>
            <div>
              <Text size='sm' fw={500} mb={4}>
                Address
              </Text>
              <TextInput
                ref={inputRef}
                placeholder='Search for an address or place...'
                {...form.getInputProps('addr_line')}
              />
              <Text size='xs' c='dimmed' mt={4}>
                Type to search for addresses using Google Places
              </Text>
            </div>

            {/* Interactive Map */}
            <div>
              <Text size='sm' fw={500} mb={4}>
                Map
              </Text>
              <Paper
                withBorder
                style={{
                  width: '100%',
                  height: 400,
                  position: 'relative',
                  overflow: 'hidden',
                }}
              >
                <Map
                  defaultZoom={12}
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
              <Text size='xs' c='dimmed' mt={4}>
                Click on the map to select a location and get coordinates
              </Text>
            </div>

            <Group grow>
              <NumberInput
                label='Latitude'
                placeholder='e.g., 40.7128'
                decimalScale={6}
                min={-90}
                max={90}
                {...form.getInputProps('lat')}
              />
              <NumberInput
                label='Longitude'
                placeholder='e.g., -74.0060'
                decimalScale={6}
                min={-180}
                max={180}
                {...form.getInputProps('lon')}
              />
            </Group>

            <Text size='xs' c='dimmed'>
              You can search for an address, click on the map, or manually enter
              coordinates.
            </Text>

            <Button type='submit' loading={isSubmitting}>
              {initialLocation ? 'Update Location' : 'Set Location'}
            </Button>
          </Stack>
        </form>
      </APIProvider>
    </Stack>
  );
}

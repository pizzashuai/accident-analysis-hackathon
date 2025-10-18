import { useState } from 'react';
import {
  TextInput,
  Button,
  Stack,
  Group,
  Text,
  NumberInput,
} from '@mantine/core';
import { useForm } from '@mantine/form';
import { useSetProjectLocation } from '~/hooks/useProjects';
import { useCustomToast } from '~/hooks/useCustomToast';

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
  const setLocation = useSetProjectLocation();
  const { showToast } = useCustomToast();

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

      <form onSubmit={form.onSubmit(handleSubmit)}>
        <Stack gap='md'>
          <TextInput
            label='Address'
            placeholder='Enter street address or location name'
            {...form.getInputProps('addr_line')}
          />

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
            You can find coordinates using Google Maps or other mapping
            services. Right-click on a location in Google Maps and select the
            coordinates.
          </Text>

          <Button type='submit' loading={isSubmitting}>
            {initialLocation ? 'Update Location' : 'Set Location'}
          </Button>
        </Stack>
      </form>
    </Stack>
  );
}

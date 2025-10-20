import { Card, Group, Text, Badge, Stack, ActionIcon } from '@mantine/core';
import {
  IconClock,
  IconGauge,
  IconRuler,
  IconArrowRight,
  IconCar,
  IconAlertTriangle,
  IconArrowUp,
  IconArrowDown,
} from '@tabler/icons-react';
import type { TimelineEvent } from '~/hooks/useLLMAnalysis';

interface TimelineCardProps {
  phase: string;
  frame: number;
  timestamp: number;
  speed_mph?: number;
  distance_m?: number;
  description: string;
  onClick: () => void;
  isActive?: boolean;
}

const getPhaseIcon = (phase: string) => {
  if (!phase) return <IconClock size={16} />;

  const phaseLower = phase.toLowerCase();
  if (phaseLower.includes('approach') || phaseLower.includes('approaching')) {
    return <IconArrowRight size={16} />;
  } else if (
    phaseLower.includes('collision') ||
    phaseLower.includes('impact') ||
    phaseLower.includes('contact')
  ) {
    return <IconAlertTriangle size={16} />;
  } else if (
    phaseLower.includes('separation') ||
    phaseLower.includes('separating')
  ) {
    return <IconArrowUp size={16} />;
  } else if (phaseLower.includes('peak') || phaseLower.includes('overlap')) {
    return <IconCar size={16} />;
  }
  return <IconClock size={16} />;
};

const getPhaseColor = (phase: string) => {
  if (!phase) return 'gray';

  const phaseLower = phase.toLowerCase();
  if (phaseLower.includes('approach') || phaseLower.includes('approaching')) {
    return 'blue';
  } else if (
    phaseLower.includes('collision') ||
    phaseLower.includes('impact') ||
    phaseLower.includes('contact')
  ) {
    return 'red';
  } else if (
    phaseLower.includes('separation') ||
    phaseLower.includes('separating')
  ) {
    return 'green';
  } else if (phaseLower.includes('peak') || phaseLower.includes('overlap')) {
    return 'orange';
  }
  return 'gray';
};

const formatTimestamp = (timestamp: number) => {
  const minutes = Math.floor(timestamp / 60);
  const seconds = Math.floor(timestamp % 60);
  const milliseconds = Math.floor((timestamp % 1) * 1000);
  return `${minutes}:${seconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(3, '0')}`;
};

export const TimelineCard = ({
  phase,
  frame,
  timestamp,
  speed_mph,
  distance_m,
  description,
  onClick,
  isActive = false,
}: TimelineCardProps) => {
  const handleClick = () => {
    console.log('TimelineCard clicked:', {
      phase,
      frame,
      timestamp,
      description,
      onClick: !!onClick,
    });
    if (onClick) {
      onClick();
    } else {
      console.warn('TimelineCard: onClick handler is not provided');
    }
  };

  return (
    <Card
      withBorder
      p='sm'
      style={{
        cursor: 'pointer',
        transition: 'all 0.2s ease',
        backgroundColor: isActive ? 'var(--mantine-color-blue-0)' : undefined,
        borderColor: isActive ? 'var(--mantine-color-blue-4)' : undefined,
      }}
      onClick={handleClick}
      onMouseEnter={(e) => {
        if (!isActive) {
          e.currentTarget.style.backgroundColor = 'var(--mantine-color-gray-0)';
        }
      }}
      onMouseLeave={(e) => {
        if (!isActive) {
          e.currentTarget.style.backgroundColor = '';
        }
      }}
    >
      <Group justify='space-between' align='flex-start'>
        <Group gap='sm' style={{ flex: 1 }}>
          <Badge
            color={getPhaseColor(phase)}
            leftSection={getPhaseIcon(phase)}
            size='sm'
            variant={isActive ? 'filled' : 'light'}
          >
            {phase || 'Unknown'}
          </Badge>

          <Stack gap={2} style={{ flex: 1 }}>
            <Text size='sm' fw={500}>
              {description}
            </Text>

            <Group gap='md'>
              <Group gap={4}>
                <IconClock size={14} />
                <Text size='xs' c='dimmed'>
                  {formatTimestamp(timestamp)}
                </Text>
              </Group>

              <Text size='xs' c='dimmed'>
                Frame {frame}
              </Text>

              {speed_mph !== undefined && (
                <Group gap={4}>
                  <IconGauge size={14} />
                  <Text size='xs' c='dimmed'>
                    {speed_mph.toFixed(1)} mph
                  </Text>
                </Group>
              )}

              {distance_m !== undefined && (
                <Group gap={4}>
                  <IconRuler size={14} />
                  <Text size='xs' c='dimmed'>
                    {(distance_m * 0.000621371).toFixed(2)} mi
                  </Text>
                </Group>
              )}
            </Group>
          </Stack>
        </Group>

        <ActionIcon variant='subtle' size='sm'>
          <IconArrowRight size={14} />
        </ActionIcon>
      </Group>
    </Card>
  );
};

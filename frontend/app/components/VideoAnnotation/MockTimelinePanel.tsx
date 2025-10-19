import { Card, Stack, Group, Text, Badge, Button, Alert } from '@mantine/core';
import {
  IconClock,
  IconAlertCircle,
  IconPlayerPlay,
} from '@tabler/icons-react';
import { TimelineCard } from './TimelineCard';
import type { TimelineEvent } from '~/hooks/useLLMAnalysis';

interface MockTimelinePanelProps {
  onSeekToTimestamp?: (timestamp: number) => void;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
}

// Mock timeline data for testing
const mockTimelineEvents: TimelineEvent[] = [
  {
    phase: 'Approach',
    frame: 80,
    timestamp: 2.667,
    speed_mph: 35.2,
    distance_m: 50.0,
    description: 'Vehicle approaches intersection at moderate speed',
  },
  {
    phase: 'Collision Impact',
    frame: 180,
    timestamp: 6.0,
    speed_mph: 28.5,
    distance_m: 15.0,
    description: 'Initial contact between vehicles occurs',
  },
  {
    phase: 'Peak Overlap',
    frame: 210,
    timestamp: 7.0,
    speed_mph: 22.1,
    distance_m: 8.0,
    description: 'Maximum overlap of vehicles during collision',
  },
  {
    phase: 'Separation',
    frame: 300,
    timestamp: 10.0,
    speed_mph: 15.3,
    distance_m: 25.0,
    description: 'Vehicles begin to separate after impact',
  },
  {
    phase: 'Post-Impact',
    frame: 450,
    timestamp: 15.0,
    speed_mph: 8.7,
    distance_m: 40.0,
    description: 'Final positions after collision sequence',
  },
];

export const MockTimelinePanel = ({
  onSeekToTimestamp,
  videoRef,
}: MockTimelinePanelProps) => {
  const handleSeekToTimestamp = (timestamp: number) => {
    console.log('MockTimelinePanel: Seeking to timestamp:', timestamp);
    if (onSeekToTimestamp) {
      onSeekToTimestamp(timestamp);
    } else {
      console.warn('MockTimelinePanel: onSeekToTimestamp not provided');
    }
  };

  const handleTestVideoRef = () => {
    if (videoRef?.current) {
      console.log('MockTimelinePanel: Video ref test:', {
        currentTime: videoRef.current.currentTime,
        duration: videoRef.current.duration,
        paused: videoRef.current.paused,
        readyState: videoRef.current.readyState,
      });

      // Test direct seeking
      console.log('MockTimelinePanel: Testing direct seek to 5 seconds');
      videoRef.current.currentTime = 5;
      videoRef.current.pause();

      setTimeout(() => {
        console.log('MockTimelinePanel: After direct seek:', {
          currentTime: videoRef.current?.currentTime,
          paused: videoRef.current?.paused,
        });
      }, 200);
    } else {
      console.warn('MockTimelinePanel: Video ref not available');
    }
  };

  return (
    <Card withBorder shadow='sm'>
      <Stack gap='md'>
        {/* Header */}
        <Group justify='space-between' align='flex-start'>
          <div>
            <Text fw={600} size='lg'>
              Mock Timeline (Testing)
            </Text>
            <Text size='sm' c='dimmed'>
              Test timeline component with mock data for debugging
            </Text>
          </div>
          <Button
            leftSection={<IconPlayerPlay size={16} />}
            onClick={handleTestVideoRef}
            variant='light'
            size='sm'
          >
            Test Video Ref
          </Button>
        </Group>

        {/* Debug Info */}
        <Alert color='blue' icon={<IconAlertCircle size={16} />}>
          <Text size='sm'>
            <strong>Debug Info:</strong>
            <br />
            onSeekToTimestamp:{' '}
            {onSeekToTimestamp ? '✅ Provided' : '❌ Missing'}
            <br />
            videoRef: {videoRef ? '✅ Provided' : '❌ Missing'}
            <br />
            Video Element:{' '}
            {videoRef?.current ? '✅ Available' : '❌ Not Available'}
          </Text>
        </Alert>

        {/* Timeline Section */}
        <Card withBorder p='sm'>
          <Group justify='space-between' mb='sm'>
            <Group gap='xs'>
              <IconClock size={16} />
              <Text fw={500}>Mock Event Timeline</Text>
              <Badge size='sm' color='blue' variant='light'>
                {mockTimelineEvents.length}
              </Badge>
            </Group>
          </Group>
          <Stack gap='sm'>
            {mockTimelineEvents.map((event, index) => (
              <TimelineCard
                key={`mock-${event.phase}-${event.frame}-${index}`}
                phase={event.phase}
                frame={event.frame}
                timestamp={event.timestamp}
                speed_mph={event.speed_mph}
                distance_m={event.distance_m}
                description={event.description}
                onClick={() => handleSeekToTimestamp(event.timestamp)}
                isActive={false}
              />
            ))}
          </Stack>
        </Card>

        {/* Instructions */}
        <Card withBorder p='sm' bg='gray.0'>
          <Text size='sm' fw={500} mb='xs'>
            Testing Instructions:
          </Text>
          <Text size='xs' c='dimmed'>
            1. Click on any timeline event above
            <br />
            2. Check browser console for debug logs
            <br />
            3. Verify video seeks to correct timestamp
            <br />
            4. Video should pause at the target frame
          </Text>
        </Card>
      </Stack>
    </Card>
  );
};

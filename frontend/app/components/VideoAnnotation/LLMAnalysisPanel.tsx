import {
  Card,
  Stack,
  Group,
  Button,
  Text,
  Badge,
  Collapse,
  ScrollArea,
  Alert,
  Loader,
  Paper,
  ActionIcon,
  Progress,
} from '@mantine/core';
import {
  IconBrain,
  IconChevronDown,
  IconChevronUp,
  IconPlayerPlay,
  IconPlayerStop,
  IconRefresh,
  IconCheck,
  IconX,
  IconAlertCircle,
  IconTool,
  IconFileText,
  IconCloud,
  IconTemperature,
  IconEye,
  IconDroplet,
  IconClock,
} from '@tabler/icons-react';
import { useLLMAnalysis } from '~/hooks/useLLMAnalysis';
import { MarkdownRenderer } from '~/components/Common/MarkdownRenderer';
import { TimelineCard } from './TimelineCard';
import type { TimelineEvent } from '~/hooks/useLLMAnalysis';

interface LLMAnalysisPanelProps {
  projectId: string;
  runId?: string;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  onSeekToTimestamp?: (timestamp: number) => void;
}

export const LLMAnalysisPanel = ({
  projectId,
  runId,
  videoRef,
  onSeekToTimestamp,
}: LLMAnalysisPanelProps) => {
  const {
    state,
    startAnalysis,
    stopAnalysis,
    resetAnalysis,
    toggleThinking,
    toggleToolCalls,
  } = useLLMAnalysis(projectId, runId);

  const getPhaseIcon = (phase: typeof state.currentPhase) => {
    switch (phase) {
      case 'thinking':
        return <IconBrain size={16} />;
      case 'tool_calls':
        return <IconTool size={16} />;
      case 'reporting':
        return <IconFileText size={16} />;
      case 'complete':
        return <IconCheck size={16} />;
      case 'error':
        return <IconX size={16} />;
      default:
        return null;
    }
  };

  const getPhaseColor = (phase: typeof state.currentPhase) => {
    switch (phase) {
      case 'thinking':
        return 'blue';
      case 'tool_calls':
        return 'orange';
      case 'reporting':
        return 'green';
      case 'complete':
        return 'green';
      case 'error':
        return 'red';
      default:
        return 'gray';
    }
  };

  const getPhaseLabel = (phase: typeof state.currentPhase) => {
    switch (phase) {
      case 'thinking':
        return 'Thinking';
      case 'tool_calls':
        return 'Using Tools';
      case 'reporting':
        return 'Generating Report';
      case 'complete':
        return 'Complete';
      case 'error':
        return 'Error';
      default:
        return 'Idle';
    }
  };

  const formatToolCallInput = (input: any) => {
    if (typeof input === 'string') {
      return input;
    }
    return JSON.stringify(input, null, 2);
  };

  const formatToolCallResult = (result: any) => {
    if (typeof result === 'string') {
      return result;
    }
    return JSON.stringify(result, null, 2);
  };

  return (
    <Card withBorder shadow='sm'>
      <Stack gap='md'>
        {/* Debug: Always visible test */}
        <Alert color='blue' icon={<IconAlertCircle size={16} />}>
          <Text size='sm'>
            LLMAnalysisPanel is rendering! Current phase: {state.currentPhase},
            Running: {state.isRunning ? 'Yes' : 'No'}
          </Text>
        </Alert>

        {/* Header */}
        <Group justify='space-between' align='flex-start'>
          <div>
            <Text fw={600} size='lg'>
              AI Accident Analysis
            </Text>
            <Text size='sm' c='dimmed'>
              Let AI analyze the accident data and generate a comprehensive
              report
            </Text>
          </div>
          <Group gap='sm'>
            {!state.isRunning ? (
              <Button
                leftSection={<IconPlayerPlay size={16} />}
                onClick={startAnalysis}
                disabled={!runId}
                color='blue'
              >
                Start Analysis
              </Button>
            ) : (
              <Button
                leftSection={<IconPlayerStop size={16} />}
                onClick={stopAnalysis}
                color='red'
                variant='outline'
              >
                Stop Analysis
              </Button>
            )}
            <Button
              leftSection={<IconRefresh size={16} />}
              onClick={resetAnalysis}
              variant='subtle'
              disabled={state.isRunning}
            >
              Reset
            </Button>
          </Group>
        </Group>

        {/* Status */}
        {state.currentPhase !== 'idle' && (
          <Group gap='sm'>
            <Badge
              color={getPhaseColor(state.currentPhase)}
              leftSection={
                <Group gap={4}>
                  <Loader size='xs' color='white' />
                </Group>
              }
              size='lg'
            >
              {getPhaseLabel(state.currentPhase)}
            </Badge>
            {state.isRunning && (
              <Progress
                value={
                  state.currentPhase === 'thinking'
                    ? 25
                    : state.currentPhase === 'tool_calls'
                      ? 50
                      : state.currentPhase === 'reporting'
                        ? 75
                        : 100
                }
                size='sm'
                style={{ flex: 1, maxWidth: 200 }}
              />
            )}
          </Group>
        )}

        {/* Collision Detection Result */}
        {state.collisionResult && (
          <Alert
            color={state.collisionResult.includes('DETECTED') ? 'red' : 'green'}
            icon={<IconAlertCircle size={16} />}
          >
            <Text size='sm' fw={600}>
              {state.collisionResult}
            </Text>
          </Alert>
        )}

        {/* Weather Data */}
        {state.weatherData && (
          <Card withBorder p='sm'>
            <Group justify='space-between' mb='sm'>
              <Group gap='xs'>
                <IconCloud size={16} />
                <Text fw={500}>Weather Conditions</Text>
              </Group>
            </Group>
            <Group gap='lg'>
              <Group gap={4}>
                <IconTemperature size={16} />
                <Text size='sm'>{state.weatherData.temperature_f}°F</Text>
              </Group>
              <Group gap={4}>
                <IconCloud size={16} />
                <Text size='sm'>{state.weatherData.condition}</Text>
              </Group>
              <Group gap={4}>
                <IconDroplet size={16} />
                <Text size='sm'>{state.weatherData.precipitation}</Text>
              </Group>
              <Group gap={4}>
                <IconEye size={16} />
                <Text size='sm'>
                  {state.weatherData.visibility_mi} mi visibility
                </Text>
              </Group>
              <Text size='sm' c='dimmed'>
                Road: {state.weatherData.road_condition}
              </Text>
            </Group>
          </Card>
        )}

        {/* Error Display */}
        {state.error && (
          <Alert color='red' icon={<IconAlertCircle size={16} />}>
            <Text size='sm'>{state.error}</Text>
          </Alert>
        )}

        {/* Thinking Section */}
        {state.thinkingContent && (
          <Card withBorder p='sm'>
            <Group justify='space-between' mb='sm'>
              <Group gap='xs'>
                <IconBrain size={16} />
                <Text fw={500}>AI Thinking Process</Text>
                {state.isRunning && <Loader size='xs' />}
              </Group>
              <ActionIcon variant='subtle' size='sm' onClick={toggleThinking}>
                {state.showThinking ? (
                  <IconChevronUp size={14} />
                ) : (
                  <IconChevronDown size={14} />
                )}
              </ActionIcon>
            </Group>
            <Collapse in={state.showThinking}>
              <ScrollArea h={200} type='auto'>
                <Text
                  size='sm'
                  style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}
                >
                  {state.thinkingContent}
                </Text>
              </ScrollArea>
            </Collapse>
          </Card>
        )}

        {/* Tool Calls Section */}
        {state.toolCalls.length > 0 && (
          <Card withBorder p='sm'>
            <Group justify='space-between' mb='sm'>
              <Group gap='xs'>
                <IconTool size={16} />
                <Text fw={500}>Tool Usage</Text>
                <Badge size='sm' color='blue' variant='light'>
                  {state.toolCalls.length}
                </Badge>
              </Group>
              <ActionIcon variant='subtle' size='sm' onClick={toggleToolCalls}>
                {state.showToolCalls ? (
                  <IconChevronUp size={14} />
                ) : (
                  <IconChevronDown size={14} />
                )}
              </ActionIcon>
            </Group>
            <Collapse in={state.showToolCalls}>
              <Stack gap='sm'>
                {state.toolCalls.map((toolCall) => (
                  <Paper key={toolCall.id} p='sm' withBorder>
                    <Group justify='space-between' mb='xs'>
                      <Group gap='xs'>
                        <Badge
                          color={
                            toolCall.status === 'completed'
                              ? 'green'
                              : toolCall.status === 'running'
                                ? 'blue'
                                : toolCall.status === 'error'
                                  ? 'red'
                                  : 'gray'
                          }
                          size='sm'
                        >
                          {toolCall.tool}
                        </Badge>
                        {toolCall.status === 'running' && <Loader size='xs' />}
                        {toolCall.status === 'completed' && (
                          <IconCheck size={14} />
                        )}
                        {toolCall.status === 'error' && <IconX size={14} />}
                      </Group>
                    </Group>
                    {toolCall.reasoning && (
                      <Text size='sm' c='blue' mb='xs' fs='italic'>
                        {toolCall.reasoning}
                      </Text>
                    )}
                    <Text size='xs' c='dimmed' mb='xs'>
                      Input: {formatToolCallInput(toolCall.input)}
                    </Text>
                    {toolCall.result && (
                      <Text size='xs' c='dimmed'>
                        Result: {formatToolCallResult(toolCall.result)}
                      </Text>
                    )}
                  </Paper>
                ))}
              </Stack>
            </Collapse>
          </Card>
        )}

        {/* Timeline Section */}
        {state.timeline && state.timeline.length > 0 && (
          <Card withBorder p='sm'>
            <Group justify='space-between' mb='sm'>
              <Group gap='xs'>
                <IconClock size={16} />
                <Text fw={500}>Event Timeline</Text>
                <Badge size='sm' color='blue' variant='light'>
                  {state.timeline.length}
                </Badge>
              </Group>
            </Group>
            <Stack gap='sm'>
              {state.timeline.map((event, index) => (
                <TimelineCard
                  key={`${event.phase}-${event.frame}-${index}`}
                  phase={event.phase}
                  frame={event.frame}
                  timestamp={event.timestamp}
                  speed_mph={event.speed_mph}
                  distance_m={event.distance_m}
                  description={event.description}
                  onClick={() => {
                    console.log('Timeline event clicked in LLMAnalysisPanel:', {
                      event,
                      onSeekToTimestamp: !!onSeekToTimestamp,
                    });
                    if (onSeekToTimestamp) {
                      onSeekToTimestamp(event.timestamp);
                    } else {
                      console.warn(
                        'onSeekToTimestamp not provided to LLMAnalysisPanel'
                      );
                    }
                  }}
                  isActive={false}
                />
              ))}
            </Stack>
          </Card>
        )}

        {/* Report Section */}
        {state.reportContent && (
          <Card withBorder p='sm'>
            <Group justify='space-between' mb='sm'>
              <Group gap='xs'>
                <IconFileText size={16} />
                <Text fw={500}>Analysis Report</Text>
                {state.isRunning && <Loader size='xs' />}
              </Group>
            </Group>
            <ScrollArea h={400} type='auto'>
              <MarkdownRenderer content={state.reportContent} size='sm' />
            </ScrollArea>
          </Card>
        )}

        {/* No Run Available */}
        {!runId && (
          <Alert color='yellow' icon={<IconAlertCircle size={16} />}>
            <Text size='sm'>
              No completed processing run available. Please complete video
              processing first.
            </Text>
          </Alert>
        )}
      </Stack>
    </Card>
  );
};

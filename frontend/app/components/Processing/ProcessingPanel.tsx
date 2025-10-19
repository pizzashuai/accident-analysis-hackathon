import { useState } from 'react';
import {
  Badge,
  Button,
  Card,
  Group,
  Progress,
  Stack,
  Table,
  Text,
  Alert,
  ActionIcon,
  Tooltip,
  Modal,
  Loader,
} from '@mantine/core';
import {
  IconPlayerPlay,
  IconDownload,
  IconRefresh,
  IconAlertCircle,
  IconCheck,
  IconX,
  IconClock,
  IconVideo,
  IconFileText,
} from '@tabler/icons-react';
import {
  useProcessingRuns,
  useStartProcessing,
  useGenerateAnnotatedVideo,
  useDownloadArtifact,
  useArtifacts,
} from '../../hooks/useProcessing';
import { useCustomToast } from '../../hooks/useCustomToast';

interface ProcessingPanelProps {
  projectId: string;
  homographySolved: boolean;
}

export function ProcessingPanel({
  projectId,
  homographySolved,
}: ProcessingPanelProps) {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [showGenerateModal, setShowGenerateModal] = useState(false);

  const { showToast } = useCustomToast();

  // API hooks
  const {
    data: runsData,
    isLoading: runsLoading,
    refetch: refetchRuns,
  } = useProcessingRuns(projectId, true); // Always enabled in ProcessingPanel
  const startProcessingMutation = useStartProcessing(projectId);
  const generateVideoMutation = useGenerateAnnotatedVideo(selectedRunId || '');
  const downloadArtifactMutation = useDownloadArtifact();

  const runs = runsData?.data || [];

  const handleStartProcessing = async () => {
    try {
      await startProcessingMutation.mutateAsync({ params: {} });
      showToast('Video processing started successfully', 'success');
    } catch (error) {
      showToast('Failed to start video processing', 'error');
    }
  };

  const handleGenerateVideo = async () => {
    if (!selectedRunId) return;

    try {
      await generateVideoMutation.mutateAsync();
      showToast('Annotated video generation started', 'success');
      setShowGenerateModal(false);
    } catch (error) {
      showToast('Failed to start video generation', 'error');
    }
  };

  const handleDownloadArtifact = async (artifactId: string) => {
    try {
      await downloadArtifactMutation.mutateAsync(artifactId);
      showToast('Download started', 'success');
    } catch (error) {
      showToast('Failed to download artifact', 'error');
    }
  };

  const getStatusBadge = (runStatus: string) => {
    const statusConfig = {
      pending: { color: 'yellow', icon: IconClock, label: 'Pending' },
      running: { color: 'blue', icon: IconRefresh, label: 'Running' },
      completed: { color: 'green', icon: IconCheck, label: 'Completed' },
      failed: { color: 'red', icon: IconX, label: 'Failed' },
    };

    const config =
      statusConfig[runStatus as keyof typeof statusConfig] ||
      statusConfig.pending;
    const Icon = config.icon;

    return (
      <Badge color={config.color} leftSection={<Icon size={12} />}>
        {config.label}
      </Badge>
    );
  };

  const getArtifactIcon = (kind: string) => {
    switch (kind) {
      case 'annotated_video':
        return <IconVideo size={16} />;
      case 'jsonl_detections':
        return <IconFileText size={16} />;
      default:
        return <IconFileText size={16} />;
    }
  };

  const getArtifactLabel = (kind: string) => {
    switch (kind) {
      case 'annotated_video':
        return 'Annotated Video';
      case 'jsonl_detections':
        return 'Detection Data (JSONL)';
      default:
        return kind;
    }
  };

  // Component to display artifacts for a run
  const RunArtifacts = ({ runId }: { runId: string }) => {
    const { data: artifactsData, isLoading: artifactsLoading } =
      useArtifacts(runId);
    const artifacts = artifactsData?.data || [];

    // Auto-refresh if there are generating artifacts
    const hasGeneratingArtifacts = artifacts.some(
      (artifact) => artifact.meta?.status === 'generating'
    );

    // Use auto-refresh if there are generating artifacts
    const { data: refreshedArtifactsData } = useArtifacts(
      runId,
      hasGeneratingArtifacts
    );
    const finalArtifacts = refreshedArtifactsData?.data || artifacts;

    if (artifactsLoading) {
      return <Loader size='xs' />;
    }

    if (finalArtifacts.length === 0) {
      return (
        <Text size='xs' c='dimmed'>
          No artifacts
        </Text>
      );
    }

    return (
      <Group gap='xs'>
        {finalArtifacts.map((artifact) => {
          const isGenerating = artifact.meta?.status === 'generating';
          const isReady =
            artifact.uri && artifact.meta?.status !== 'generating';
          const hasError =
            artifact.meta?.status === 'failed' || artifact.meta?.error;

          const getStatusColor = () => {
            if (hasError) return 'red';
            if (isGenerating) return 'yellow';
            if (isReady) return 'green';
            return 'gray';
          };

          const getStatusLabel = () => {
            if (hasError)
              return `Error: ${artifact.meta?.error || 'Failed to generate'}`;
            if (isGenerating) return 'Generating...';
            if (isReady) return `Download ${getArtifactLabel(artifact.kind)}`;
            return 'Not ready';
          };

          const getFileSize = () => {
            const sizeBytes =
              artifact.meta?.video_size_bytes || artifact.meta?.file_size_bytes;
            if (!sizeBytes || typeof sizeBytes !== 'number') return '';

            const sizeMB = (sizeBytes / (1024 * 1024)).toFixed(1);
            return ` (${sizeMB} MB)`;
          };

          return (
            <Tooltip
              key={artifact.id}
              label={`${getStatusLabel()}${getFileSize()}`}
            >
              <ActionIcon
                variant='light'
                color={getStatusColor()}
                size='sm'
                onClick={() => {
                  if (isReady) {
                    handleDownloadArtifact(artifact.id);
                  }
                }}
                disabled={!isReady}
                loading={downloadArtifactMutation.isPending}
              >
                {isGenerating ? (
                  <Loader size={12} />
                ) : hasError ? (
                  <IconX size={12} />
                ) : (
                  getArtifactIcon(artifact.kind)
                )}
              </ActionIcon>
            </Tooltip>
          );
        })}
      </Group>
    );
  };

  // Helper to check if a run has generating artifacts
  const RunHasGeneratingArtifacts = ({
    runId,
    children,
  }: {
    runId: string;
    children: (hasGenerating: boolean) => React.ReactNode;
  }) => {
    const { data: artifactsData } = useArtifacts(runId);
    const artifacts = artifactsData?.data || [];
    const hasGenerating = artifacts.some(
      (artifact) => artifact.meta?.status === 'generating'
    );
    return <>{children(hasGenerating)}</>;
  };

  if (runsLoading) {
    return (
      <Card withBorder shadow='sm'>
        <Group justify='center' p='xl'>
          <Loader size='md' />
          <Text>Loading processing runs...</Text>
        </Group>
      </Card>
    );
  }

  return (
    <Stack gap='md' data-processing-panel>
      {/* Start Processing Section */}
      <Card withBorder shadow='sm'>
        <Stack gap='md'>
          <Group justify='space-between'>
            <div>
              <Text fw={600} size='lg'>
                Video Processing
              </Text>
              <Text size='sm' c='dimmed'>
                Run YOLO detection and ByteTrack tracking on your video
              </Text>
            </div>
            <Button
              leftSection={<IconPlayerPlay size={16} />}
              onClick={handleStartProcessing}
              loading={startProcessingMutation.isPending}
              disabled={!homographySolved}
            >
              Run Analysis
            </Button>
          </Group>

          {!homographySolved && (
            <Alert icon={<IconAlertCircle size={16} />} color='yellow'>
              Homography must be solved before processing can start
            </Alert>
          )}

          {(() => {
            // Find the currently running run to show progress
            const activeRun = runs.find(
              (run) => run.status === 'running' || run.status === 'pending'
            );
            if (
              activeRun &&
              activeRun.progress &&
              Object.keys(activeRun.progress).length > 0
            ) {
              const progress = activeRun.progress as any;
              return (
                <Stack gap='xs'>
                  <Group justify='space-between'>
                    <Text size='sm' fw={500}>
                      {progress.stage
                        ?.replace('_', ' ')
                        .replace(/\b\w/g, (l: string) => l.toUpperCase()) ||
                        'Processing'}
                    </Text>
                    <Text size='sm' c='dimmed'>
                      {progress.percent || 0}%
                    </Text>
                  </Group>
                  <Progress value={progress.percent || 0} size='sm' />
                  <Text size='xs' c='dimmed'>
                    {progress.message || 'Processing video...'}
                  </Text>
                </Stack>
              );
            }
            return null;
          })()}
        </Stack>
      </Card>

      {/* Processing Runs Table */}
      <Card withBorder shadow='sm'>
        <Stack gap='md'>
          <Group justify='space-between'>
            <Text fw={600} size='lg'>
              Processing Runs
            </Text>
            <ActionIcon variant='light' onClick={() => refetchRuns()}>
              <IconRefresh size={16} />
            </ActionIcon>
          </Group>

          {runs.length === 0 ? (
            <Text c='dimmed' ta='center' py='xl'>
              No processing runs yet. Click "Run Analysis" to start processing
              your video.
            </Text>
          ) : (
            <Table>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Status</Table.Th>
                  <Table.Th>Started</Table.Th>
                  <Table.Th>Duration</Table.Th>
                  <Table.Th>Artifacts</Table.Th>
                  <Table.Th>Actions</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {runs.map((run) => (
                  <Table.Tr key={run.id}>
                    <Table.Td>
                      {getStatusBadge(run.status)}
                      {(run.status === 'running' || run.status === 'pending') &&
                        run.progress &&
                        Object.keys(run.progress).length > 0 && (
                          <Progress
                            value={(run.progress as any).percent || 0}
                            size='xs'
                            mt='xs'
                          />
                        )}
                    </Table.Td>
                    <Table.Td>
                      <Text size='sm'>
                        {new Date(run.started_at).toLocaleString()}
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      <Text size='sm'>
                        {run.finished_at
                          ? `${Math.round((new Date(run.finished_at).getTime() - new Date(run.started_at).getTime()) / 1000)}s`
                          : 'Running...'}
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      <RunArtifacts runId={run.id} />
                    </Table.Td>
                    <Table.Td>
                      <Group gap='xs'>
                        {run.status === 'completed' && (
                          <RunHasGeneratingArtifacts runId={run.id}>
                            {(hasGenerating) => (
                              <Tooltip
                                label={
                                  hasGenerating
                                    ? 'Video is being generated...'
                                    : 'Generate Annotated Video'
                                }
                              >
                                <ActionIcon
                                  variant='light'
                                  color={hasGenerating ? 'yellow' : 'blue'}
                                  onClick={() => {
                                    if (!hasGenerating) {
                                      setSelectedRunId(run.id);
                                      setShowGenerateModal(true);
                                    }
                                  }}
                                  disabled={hasGenerating}
                                >
                                  {hasGenerating ? (
                                    <Loader size={16} />
                                  ) : (
                                    <IconVideo size={16} />
                                  )}
                                </ActionIcon>
                              </Tooltip>
                            )}
                          </RunHasGeneratingArtifacts>
                        )}
                        {run.status === 'failed' && run.error_message && (
                          <Tooltip label={run.error_message}>
                            <ActionIcon variant='light' color='red'>
                              <IconAlertCircle size={16} />
                            </ActionIcon>
                          </Tooltip>
                        )}
                      </Group>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          )}
        </Stack>
      </Card>

      {/* Generate Video Modal */}
      <Modal
        opened={showGenerateModal}
        onClose={() => setShowGenerateModal(false)}
        title='Generate Annotated Video'
        centered
      >
        <Stack gap='md'>
          <Text>
            This will generate a pre-rendered video with bounding boxes and
            speed labels overlaid. The process may take a few minutes.
          </Text>
          <Group justify='flex-end'>
            <Button variant='light' onClick={() => setShowGenerateModal(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleGenerateVideo}
              loading={generateVideoMutation.isPending}
            >
              Generate Video
            </Button>
          </Group>
        </Stack>
      </Modal>
    </Stack>
  );
}

import { useState, useCallback } from 'react';
import {
  Select,
  Group,
  Text,
  Badge,
  Button,
  Stack,
  Loader,
  Alert,
} from '@mantine/core';
import { IconDownload, IconAlertCircle } from '@tabler/icons-react';
import {
  useProjectJsonlArtifacts,
  useArtifactDownloadUrl,
} from '../../hooks/useProcessing';

interface JsonlFileSelectorProps {
  projectId: string;
  selectedArtifactId?: string | null;
  onArtifactSelect: (artifactId: string | null) => void;
  disabled?: boolean;
}

export const JsonlFileSelector = ({
  projectId,
  selectedArtifactId,
  onArtifactSelect,
  disabled = false,
}: JsonlFileSelectorProps) => {
  const [downloadingArtifactId, setDownloadingArtifactId] = useState<
    string | null
  >(null);

  const {
    data: artifacts,
    isLoading,
    error,
  } = useProjectJsonlArtifacts(projectId);
  const { data: downloadUrl } = useArtifactDownloadUrl(
    selectedArtifactId || '',
    !!selectedArtifactId
  );

  const handleDownload = useCallback(async (artifactId: string) => {
    setDownloadingArtifactId(artifactId);
    try {
      const { ProjectsService } = await import('../../client/sdk.gen');
      const response = await ProjectsService.getArtifactDownloadUrl({
        artifactId,
      });

      // Create a temporary link to trigger download
      const link = document.createElement('a');
      link.href = (response as { url: string }).url;
      link.download = ''; // Let the server determine the filename
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Failed to download artifact:', error);
    } finally {
      setDownloadingArtifactId(null);
    }
  }, []);

  const formatArtifactLabel = (artifact: any) => {
    const runDate = new Date(artifact.runStartedAt).toLocaleDateString();
    const runTime = new Date(artifact.runStartedAt).toLocaleTimeString();
    return `Run ${artifact.runId.slice(0, 8)} - ${runDate} ${runTime}`;
  };

  if (isLoading) {
    return (
      <Group gap='xs'>
        <Loader size='sm' />
        <Text size='sm' c='dimmed'>
          Loading available JSONL files...
        </Text>
      </Group>
    );
  }

  if (error) {
    return (
      <Alert icon={<IconAlertCircle size={16} />} color='red' variant='light'>
        <Text size='sm'>Failed to load JSONL files. Please try again.</Text>
      </Alert>
    );
  }

  if (!artifacts || artifacts.length === 0) {
    return (
      <Alert icon={<IconAlertCircle size={16} />} color='blue' variant='light'>
        <Text size='sm'>
          No JSONL detection files found. Run video processing to generate
          detection data.
        </Text>
      </Alert>
    );
  }

  return (
    <Stack gap='sm'>
      <Group justify='space-between' align='flex-end'>
        <Select
          label='Select JSONL Detection File'
          placeholder='Choose a processing run...'
          value={selectedArtifactId || null}
          onChange={onArtifactSelect}
          data={artifacts.map((artifact: any) => ({
            value: artifact.id,
            label: formatArtifactLabel(artifact),
          }))}
          disabled={disabled}
          style={{ flex: 1 }}
          clearable
        />

        {selectedArtifactId && (
          <Button
            variant='light'
            size='sm'
            leftSection={<IconDownload size={16} />}
            onClick={() => handleDownload(selectedArtifactId)}
            loading={downloadingArtifactId === selectedArtifactId}
            disabled={disabled}
          >
            Download
          </Button>
        )}
      </Group>

      {selectedArtifactId && (
        <Group gap='xs'>
          <Badge color='green' variant='light' size='sm'>
            JSONL Selected
          </Badge>
          <Text size='xs' c='dimmed'>
            Detection data will be loaded from the selected file
          </Text>
        </Group>
      )}
    </Stack>
  );
};

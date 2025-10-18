import { useState } from 'react';
import { Button, Text, Group, Progress, Stack } from '@mantine/core';
import { Dropzone } from '@mantine/dropzone';
import { IconUpload, IconX, IconFile } from '@tabler/icons-react';
import { useUploadVideo } from '~/hooks/useProjects';
import { useCustomToast } from '~/hooks/useCustomToast';

interface VideoUploadProps {
  projectId: string;
  onUploadComplete?: () => void;
}

export function VideoUpload({ projectId, onUploadComplete }: VideoUploadProps) {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const uploadVideo = useUploadVideo();
  const { showToast } = useCustomToast();

  const handleDrop = async (files: File[]) => {
    const file = files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('video/')) {
      showToast('Please select a video file', 'error');
      return;
    }

    // Validate file size (max 100MB)
    const maxSize = 100 * 1024 * 1024; // 100MB
    if (file.size > maxSize) {
      showToast('File size must be less than 100MB', 'error');
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    try {
      // Simulate progress for better UX
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return prev + 10;
        });
      }, 200);

      await uploadVideo.mutateAsync({ projectId, file });

      clearInterval(progressInterval);
      setUploadProgress(100);

      showToast('Video uploaded successfully', 'success');
      onUploadComplete?.();
    } catch (error) {
      showToast('Failed to upload video', 'error');
    } finally {
      setUploading(false);
      setTimeout(() => setUploadProgress(0), 1000);
    }
  };

  return (
    <Stack gap='md'>
      <Text size='sm' fw={500}>
        Upload Video
      </Text>

      <Dropzone
        onDrop={handleDrop}
        onReject={() => showToast('Invalid file type', 'error')}
        maxSize={100 * 1024 * 1024} // 100MB
        accept={{
          'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
        }}
        loading={uploading}
        disabled={uploading}
      >
        <Group
          justify='center'
          gap='xl'
          mih={220}
          style={{ pointerEvents: 'none' }}
        >
          <Dropzone.Accept>
            <IconUpload size={52} stroke={1.5} />
          </Dropzone.Accept>
          <Dropzone.Reject>
            <IconX size={52} stroke={1.5} />
          </Dropzone.Reject>
          <Dropzone.Idle>
            <IconFile size={52} stroke={1.5} />
          </Dropzone.Idle>

          <div>
            <Text size='xl' inline>
              Drag video file here or click to select
            </Text>
            <Text size='sm' c='dimmed' inline mt={7}>
              Supports MP4, AVI, MOV, MKV, WebM (max 100MB)
            </Text>
          </div>
        </Group>
      </Dropzone>

      {uploading && (
        <Stack gap='xs'>
          <Text size='sm'>Uploading video...</Text>
          <Progress value={uploadProgress} size='sm' />
        </Stack>
      )}
    </Stack>
  );
}

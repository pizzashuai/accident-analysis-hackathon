import { useState, useEffect } from 'react';
import {
  Button,
  Text,
  Group,
  Progress,
  Stack,
  Alert,
  Divider,
} from '@mantine/core';
import { DateTimePicker } from '@mantine/dates';
import { Dropzone } from '@mantine/dropzone';
import {
  IconUpload,
  IconX,
  IconFile,
  IconCalendar,
  IconCheck,
  IconEdit,
} from '@tabler/icons-react';
import { useUploadVideo, useUpdateVideoStartTime } from '~/hooks/useProjects';
import { useCustomToast } from '~/hooks/useCustomToast';
import type { ProjectPublic } from '~/client';

interface VideoUploadProps {
  projectId: string;
  project?: ProjectPublic;
  onUploadComplete?: () => void;
}

export function VideoUpload({
  projectId,
  project,
  onUploadComplete,
}: VideoUploadProps) {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [videoStartTime, setVideoStartTime] = useState<string | null>(null);
  const [isUpdatingStartTime, setIsUpdatingStartTime] = useState(false);
  const uploadVideo = useUploadVideo();
  const updateVideoStartTime = useUpdateVideoStartTime();
  const { showToast } = useCustomToast();

  // Initialize video start time from existing project data
  useEffect(() => {
    if (project?.video?.video_start_time) {
      setVideoStartTime(project.video.video_start_time);
    }
  }, [project?.video?.video_start_time]);

  const handleUpdateVideoStartTime = async () => {
    if (!project?.video?.id) {
      showToast('No video found to update', 'error');
      return;
    }

    setIsUpdatingStartTime(true);
    try {
      await updateVideoStartTime.mutateAsync({
        projectId,
        mediaAssetId: project.video.id,
        videoStartTime: videoStartTime || undefined,
      });
      showToast('Video start time updated successfully', 'success');
    } catch (error) {
      showToast('Failed to update video start time', 'error');
    } finally {
      setIsUpdatingStartTime(false);
    }
  };

  const getVideoDuration = (file: File): Promise<number> => {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video');
      video.preload = 'metadata';

      video.onloadedmetadata = () => {
        window.URL.revokeObjectURL(video.src);
        resolve(video.duration);
      };

      video.onerror = () => {
        window.URL.revokeObjectURL(video.src);
        reject(new Error('Failed to load video metadata'));
      };

      video.src = URL.createObjectURL(file);
    });
  };

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

    // Validate video duration (max 5 seconds)
    try {
      const duration = await getVideoDuration(file);
      if (duration > 5) {
        showToast(
          `Video duration (${duration.toFixed(1)}s) exceeds 5 second limit`,
          'error'
        );
        return;
      }
    } catch (error) {
      showToast('Failed to validate video duration', 'error');
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

      await uploadVideo.mutateAsync({
        projectId,
        file,
        videoStartTime: videoStartTime || '',
      });

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

  const hasVideo = !!project?.video;
  const hasVideoStartTime = !!project?.video?.video_start_time;
  const isStartTimeChanged =
    videoStartTime !== project?.video?.video_start_time;

  return (
    <Stack gap='md'>
      <Text size='sm' fw={500}>
        Upload Video
      </Text>

      <Alert color='blue' icon={<IconCalendar size={16} />}>
        <Text size='sm'>
          Optional: Set the real-world start time of the video to calculate
          accurate event timestamps in the JSONL output.
        </Text>
      </Alert>

      {/* Video Start Time Section */}
      <Stack gap='sm'>
        <Group justify='space-between' align='center'>
          <Text size='sm' fw={500}>
            Video Start Time
          </Text>
          {hasVideoStartTime && (
            <Group gap='xs'>
              <IconCheck size={16} color='var(--mantine-color-green-6)' />
              <Text size='xs' c='green'>
                Set
              </Text>
            </Group>
          )}
        </Group>

        <DateTimePicker
          label={
            hasVideo ? 'Update Video Start Time' : 'Video Start Time (Optional)'
          }
          placeholder='Pick date and time'
          value={videoStartTime}
          onChange={setVideoStartTime}
          leftSection={<IconCalendar size={16} />}
          description={
            hasVideo
              ? 'Update the real-world start time of the uploaded video'
              : 'Set the real-world start time of the video for accurate event timestamps'
          }
          disabled={uploading || isUpdatingStartTime}
          clearable
        />

        {hasVideo && (
          <Group gap='sm'>
            <Button
              size='xs'
              variant='light'
              onClick={handleUpdateVideoStartTime}
              loading={isUpdatingStartTime}
              disabled={!isStartTimeChanged}
              leftSection={<IconEdit size={14} />}
            >
              {isStartTimeChanged ? 'Update Start Time' : 'No Changes'}
            </Button>
            {isStartTimeChanged && (
              <Text size='xs' c='orange'>
                You have unsaved changes
              </Text>
            )}
          </Group>
        )}
      </Stack>

      {hasVideo && <Divider />}

      {/* Video Upload Section */}
      <Stack gap='sm'>
        <Text size='sm' fw={500}>
          {hasVideo ? 'Replace Video' : 'Upload Video'}
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
                {hasVideo
                  ? 'Drag new video here or click to replace'
                  : 'Drag video file here or click to select'}
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
    </Stack>
  );
}

import type { Route } from './+types/projects.$projectId';
import { useState, useEffect } from 'react';
import {
  Container,
  Title,
  Text,
  Stack,
  Group,
  Button,
  Card,
  Badge,
  Grid,
  Tabs,
  Image,
  Loader,
} from '@mantine/core';
import {
  IconArrowLeft,
  IconEdit,
  IconTrash,
  IconMapPin,
  IconCalendar,
  IconVideo,
  IconPhoto,
} from '@tabler/icons-react';
import { Link, useNavigate } from 'react-router';
import { CreateProjectModal } from '~/components/Projects/CreateProjectModal';
import { VideoUpload } from '~/components/Projects/VideoUpload';
import { LocationPicker } from '~/components/Projects/LocationPicker';
import { HomographyPicker } from '~/homography';
import { VideoAnnotationViewer } from '~/components/VideoAnnotation/VideoAnnotationViewer';
import {
  useProject,
  useDeleteProject,
  useMediaPresignedUrl,
} from '~/hooks/useProjects';
import { useCustomToast } from '~/hooks/useCustomToast';

export function meta({ params }: Route.MetaArgs) {
  return [
    { title: `Project ${params.projectId} - Accident Analysis` },
    { name: 'description', content: 'View and manage project details' },
  ];
}

export default function ProjectDetail({ params }: Route.ComponentProps) {
  const navigate = useNavigate();
  const [editModalOpened, setEditModalOpened] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [screenshotUrl, setScreenshotUrl] = useState<string | null>(null);
  const [loadingVideoUrl, setLoadingVideoUrl] = useState(false);
  const [loadingScreenshot, setLoadingScreenshot] = useState(false);
  const { data: project, isLoading, error } = useProject(params.projectId);
  const deleteProject = useDeleteProject();
  const getMediaPresignedUrl = useMediaPresignedUrl();
  const { showToast } = useCustomToast();

  const handleDelete = async () => {
    if (
      !confirm(
        'Are you sure you want to delete this project? This action cannot be undone.'
      )
    ) {
      return;
    }

    setIsDeleting(true);
    try {
      await deleteProject.mutateAsync(params.projectId);
      showToast('Project deleted successfully', 'success');
      navigate('/projects');
    } catch (error) {
      showToast('Failed to delete project', 'error');
    } finally {
      setIsDeleting(false);
    }
  };

  const handleEdit = () => {
    setEditModalOpened(true);
  };

  const handleUploadComplete = () => {
    // Refresh project data
    window.location.reload();
  };

  const handleLocationSet = () => {
    // Refresh project data
    window.location.reload();
  };

  // Fetch presigned URL for video
  const fetchVideoUrl = async () => {
    if (!project?.video) return;

    setLoadingVideoUrl(true);
    try {
      const response = await getMediaPresignedUrl.mutateAsync({
        projectId: params.projectId,
        mediaAssetId: project.video.id,
      });
      setVideoUrl((response as { url: string }).url);
    } catch (error) {
      console.error('Failed to fetch video URL:', error);
      showToast('Failed to load video', 'error');
    } finally {
      setLoadingVideoUrl(false);
    }
  };

  // Fetch screenshot URL
  const fetchScreenshotUrl = async () => {
    if (!project?.media_assets) return;

    const screenshot = project.media_assets.find(
      (asset) => asset.kind === 'image'
    );
    if (!screenshot) return;

    setLoadingScreenshot(true);
    try {
      const response = await getMediaPresignedUrl.mutateAsync({
        projectId: params.projectId,
        mediaAssetId: screenshot.id,
      });
      setScreenshotUrl((response as { url: string }).url);
    } catch (error) {
      console.error('Failed to fetch screenshot URL:', error);
    } finally {
      setLoadingScreenshot(false);
    }
  };

  // Fetch URLs when project data changes
  useEffect(() => {
    if (project?.video) {
      fetchVideoUrl();
    }
    if (project?.media_assets) {
      fetchScreenshotUrl();
    }
  }, [project]);

  if (isLoading) {
    return (
      <Container>
        <Title order={1} mb='md'>
          Loading project...
        </Title>
      </Container>
    );
  }

  if (error || !project) {
    return (
      <Container>
        <Title order={1} mb='md'>
          Project Not Found
        </Title>
        <Text c='red'>The requested project could not be found.</Text>
        <Button component={Link} to='/projects' mt='md'>
          Back to Projects
        </Button>
      </Container>
    );
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  return (
    <Container>
      <Stack gap='lg'>
        {/* Header */}
        <Group justify='space-between' align='flex-start'>
          <Group gap='md'>
            <Button
              component={Link}
              to='/projects'
              variant='outline'
              leftSection={<IconArrowLeft size={16} />}
            >
              Back to Projects
            </Button>
            <div>
              <Title order={1} mb='xs'>
                {project.title}
              </Title>
              <Text c='dimmed'>
                Created on {formatDate(project.created_at)}
              </Text>
            </div>
          </Group>
          <Group gap='sm'>
            <Button
              variant='outline'
              leftSection={<IconEdit size={16} />}
              onClick={handleEdit}
            >
              Edit
            </Button>
            <Button
              color='red'
              variant='outline'
              leftSection={<IconTrash size={16} />}
              onClick={handleDelete}
              loading={isDeleting}
            >
              Delete
            </Button>
          </Group>
        </Group>

        {/* Project Info */}
        <Card withBorder>
          <Stack gap='md'>
            <Group gap='xs'>
              <IconCalendar size={16} />
              <Text fw={500}>Project Information</Text>
            </Group>

            {project.description && <Text>{project.description}</Text>}

            <Group gap='xs'>
              {project.video && (
                <Badge
                  color='green'
                  variant='light'
                  leftSection={<IconVideo size={12} />}
                >
                  Video uploaded
                </Badge>
              )}
              {project.location && (
                <Badge
                  color='blue'
                  variant='light'
                  leftSection={<IconMapPin size={12} />}
                >
                  Location set
                </Badge>
              )}
              {project.homography_session && (
                <Badge
                  color={
                    project.homography_session.status === 'solved'
                      ? 'green'
                      : 'yellow'
                  }
                  variant='light'
                  leftSection={<IconPhoto size={12} />}
                >
                  Homography{' '}
                  {project.homography_session.status === 'solved'
                    ? 'solved'
                    : 'configured'}
                </Badge>
              )}
            </Group>
          </Stack>
        </Card>

        {/* Tabs for different sections */}
        <Tabs defaultValue='overview'>
          <Tabs.List>
            <Tabs.Tab value='overview'>Overview</Tabs.Tab>
            <Tabs.Tab value='video'>Video</Tabs.Tab>
            <Tabs.Tab value='location'>Location</Tabs.Tab>
            <Tabs.Tab value='homography'>Homography</Tabs.Tab>
          </Tabs.List>

          <Tabs.Panel value='overview' pt='md'>
            <Stack gap='md'>
              <Text size='lg' fw={500}>
                Project Overview
              </Text>

              <Grid>
                <Grid.Col span={{ base: 12, md: 4 }}>
                  <Card withBorder>
                    <Stack gap='sm'>
                      <Text fw={500}>Video Status</Text>
                      {project.video ? (
                        <Group gap='xs'>
                          <IconVideo size={16} color='green' />
                          <Text size='sm'>Video uploaded successfully</Text>
                        </Group>
                      ) : (
                        <Group gap='xs'>
                          <IconVideo size={16} color='gray' />
                          <Text size='sm' c='dimmed'>
                            No video uploaded
                          </Text>
                        </Group>
                      )}
                    </Stack>
                  </Card>
                </Grid.Col>

                <Grid.Col span={{ base: 12, md: 4 }}>
                  <Card withBorder>
                    <Stack gap='sm'>
                      <Text fw={500}>Screenshot Status</Text>
                      {project.media_assets?.find(
                        (asset) => asset.kind === 'image'
                      ) ? (
                        <Group gap='xs'>
                          <IconPhoto size={16} color='green' />
                          <Text size='sm'>Screenshot available</Text>
                        </Group>
                      ) : project.video?.is_processing ? (
                        <Group gap='xs'>
                          <Loader size='xs' />
                          <Text size='sm' c='blue'>
                            Extracting frame...
                          </Text>
                        </Group>
                      ) : (
                        <Group gap='xs'>
                          <IconPhoto size={16} color='gray' />
                          <Text size='sm' c='dimmed'>
                            No screenshot yet
                          </Text>
                        </Group>
                      )}
                    </Stack>
                  </Card>
                </Grid.Col>

                <Grid.Col span={{ base: 12, md: 4 }}>
                  <Card withBorder>
                    <Stack gap='sm'>
                      <Text fw={500}>Location Status</Text>
                      {project.location ? (
                        <Group gap='xs'>
                          <IconMapPin size={16} color='blue' />
                          <Text size='sm'>
                            {project.location.addr_line || 'Coordinates set'}
                          </Text>
                        </Group>
                      ) : (
                        <Group gap='xs'>
                          <IconMapPin size={16} color='gray' />
                          <Text size='sm' c='dimmed'>
                            No location set
                          </Text>
                        </Group>
                      )}
                    </Stack>
                  </Card>
                </Grid.Col>
              </Grid>

              {/* Screenshot Display */}
              {project.media_assets?.find(
                (asset) => asset.kind === 'image'
              ) && (
                <Card withBorder>
                  <Stack gap='md'>
                    <Text fw={500}>Video Screenshot</Text>
                    {loadingScreenshot ? (
                      <Group justify='center' p='xl'>
                        <Loader size='md' />
                        <Text size='sm' c='dimmed'>
                          Loading screenshot...
                        </Text>
                      </Group>
                    ) : screenshotUrl ? (
                      <Image
                        src={screenshotUrl}
                        alt='Video screenshot'
                        style={{ maxWidth: '400px', maxHeight: '300px' }}
                        onError={() => {
                          showToast(
                            'Screenshot failed to load, refreshing...',
                            'error'
                          );
                          fetchScreenshotUrl();
                        }}
                      />
                    ) : (
                      <Group justify='center' p='xl'>
                        <Text size='sm' c='dimmed'>
                          Failed to load screenshot
                        </Text>
                        <Button
                          size='xs'
                          variant='outline'
                          onClick={fetchScreenshotUrl}
                        >
                          Retry
                        </Button>
                      </Group>
                    )}
                    <Text size='sm' c='dimmed'>
                      First frame extracted from video for homography mapping
                    </Text>
                  </Stack>
                </Card>
              )}
            </Stack>
          </Tabs.Panel>

          <Tabs.Panel value='video' pt='md'>
            <Stack gap='md'>
              <Text size='lg' fw={500}>
                Video Management
              </Text>

              {project.video ? (
                <Card withBorder>
                  <Stack gap='md'>
                    <Text fw={500}>Current Video</Text>
                    {loadingVideoUrl ? (
                      <Group justify='center' p='xl'>
                        <Loader size='md' />
                        <Text size='sm' c='dimmed'>
                          Loading video...
                        </Text>
                      </Group>
                    ) : videoUrl ? (
                      <video
                        controls
                        style={{ width: '100%', maxWidth: '600px' }}
                        src={videoUrl}
                        onError={() => {
                          showToast(
                            'Video failed to load, refreshing...',
                            'error'
                          );
                          fetchVideoUrl();
                        }}
                      >
                        Your browser does not support the video tag.
                      </video>
                    ) : (
                      <Group justify='center' p='xl'>
                        <Text size='sm' c='dimmed'>
                          Failed to load video
                        </Text>
                        <Button
                          size='xs'
                          variant='outline'
                          onClick={fetchVideoUrl}
                        >
                          Retry
                        </Button>
                      </Group>
                    )}

                    {/* Show processing status */}
                    {project.video.is_processing && (
                      <Group gap='xs'>
                        <Loader size='xs' />
                        <Text size='sm' c='blue'>
                          Processing video frame...
                        </Text>
                      </Group>
                    )}

                    {project.video.processing_error && (
                      <Text size='sm' c='red'>
                        Processing error: {project.video.processing_error}
                      </Text>
                    )}

                    <Text size='sm' c='dimmed'>
                      Upload a new video to replace the current one.
                    </Text>
                  </Stack>
                </Card>
              ) : (
                <Card withBorder>
                  <Text size='sm' c='dimmed' mb='md'>
                    No video uploaded yet. Upload a video to get started with
                    accident analysis.
                  </Text>
                </Card>
              )}

              <VideoUpload
                projectId={params.projectId}
                onUploadComplete={handleUploadComplete}
              />

              {videoUrl && (
                <VideoAnnotationViewer
                  videoUrl={videoUrl}
                />
              )}
            </Stack>
          </Tabs.Panel>

          <Tabs.Panel value='location' pt='md'>
            <Stack gap='md'>
              <Text size='lg' fw={500}>
                Location Management
              </Text>

              {project.location && (
                <Card withBorder>
                  <Stack gap='sm'>
                    <Text fw={500}>Current Location</Text>
                    {project.location.addr_line && (
                      <Text>{project.location.addr_line}</Text>
                    )}
                    {project.location.lat && project.location.lon && (
                      <Text size='sm' c='dimmed'>
                        Coordinates: {project.location.lat.toFixed(6)},{' '}
                        {project.location.lon.toFixed(6)}
                      </Text>
                    )}
                  </Stack>
                </Card>
              )}

              <LocationPicker
                projectId={params.projectId}
                initialLocation={
                  project.location
                    ? {
                        addr_line: project.location.addr_line || undefined,
                        lat: project.location.lat || undefined,
                        lon: project.location.lon || undefined,
                      }
                    : null
                }
                onLocationSet={handleLocationSet}
              />
            </Stack>
          </Tabs.Panel>

          <Tabs.Panel value='homography' pt='md'>
            <Stack gap='md'>
              <Text size='lg' fw={500}>
                Homography Configuration
              </Text>
              <Text size='sm' c='dimmed'>
                Configure point correspondences between your CCTV video and map
                coordinates for accurate speed calculation.
              </Text>

              <HomographyPicker
                projectId={params.projectId}
                existingSession={project.homography_session}
              />
            </Stack>
          </Tabs.Panel>
        </Tabs>
      </Stack>

      <CreateProjectModal
        opened={editModalOpened}
        onClose={() => setEditModalOpened(false)}
        editingProject={
          project
            ? {
                id: project.id,
                title: project.title,
                description: project.description || undefined,
              }
            : null
        }
      />
    </Container>
  );
}

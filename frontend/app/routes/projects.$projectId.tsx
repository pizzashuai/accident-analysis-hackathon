import type { Route } from './+types/projects.$projectId';
import { useState, useEffect, useRef } from 'react';
import {
  Container,
  Title,
  Text,
  Stack,
  Group,
  Button,
  Card,
  Badge,
  ActionIcon,
  Stepper,
  Paper,
  Alert,
  Box,
} from '@mantine/core';
import {
  IconArrowLeft,
  IconEdit,
  IconTrash,
  IconMapPin,
  IconCalendar,
  IconVideo,
  IconPhoto,
  IconCpu,
  IconCheck,
  IconEye,
  IconAlertCircle,
} from '@tabler/icons-react';
import { Link, useNavigate } from 'react-router';
import { CreateProjectModal } from '~/components/Projects/CreateProjectModal';
import { ProjectWorkflow } from '~/components/Projects/ProjectWorkflow';
import { VideoAnnotationViewer } from '~/components/VideoAnnotation/VideoAnnotationViewer';
import { LLMAnalysisPanel } from '~/components/VideoAnnotation/LLMAnalysisPanel';
import {
  useProject,
  useDeleteProject,
  useMediaPresignedUrl,
} from '~/hooks/useProjects';
import { useProcessingRuns } from '~/hooks/useProcessing';
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
  const [activeStep, setActiveStep] = useState(0);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isLoadingVideoUrl, setIsLoadingVideoUrl] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const {
    data: project,
    isLoading,
    error,
    refetch,
  } = useProject(params.projectId);
  const { data: processingRuns } = useProcessingRuns(params.projectId, true);
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

  const handleRefresh = () => {
    refetch();
  };

  const handleSeekToTimestamp = (timestamp: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = timestamp;
      videoRef.current.pause();
    }
  };

  // Fetch video URL when project data changes
  useEffect(() => {
    const fetchVideoUrl = async () => {
      if (!project?.video) {
        setVideoUrl(null);
        return;
      }

      // Use presigned URL if available
      if (project.video.presigned_url) {
        setVideoUrl(project.video.presigned_url);
        return;
      }

      // Try to fetch presigned URL if not available
      if (project.video.id) {
        setIsLoadingVideoUrl(true);
        try {
          const response = await getMediaPresignedUrl.mutateAsync({
            projectId: params.projectId,
            mediaAssetId: project.video.id,
          });
          const url = (response as { url: string }).url;
          setVideoUrl(url);
        } catch (error) {
          console.error('Failed to fetch presigned URL:', error);
          // Fallback to URI
          setVideoUrl(project.video.uri);
          showToast('Using fallback video URL', 'info');
        } finally {
          setIsLoadingVideoUrl(false);
        }
      } else {
        // Fallback to URI
        setVideoUrl(project.video.uri);
      }
    };

    fetchVideoUrl();
  }, [project, params.projectId, getMediaPresignedUrl, showToast]);

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

  // Helper function to check if workflow is complete
  const isWorkflowComplete = () => {
    if (!project) return false;

    const hasVideo =
      project.video &&
      !project.video.processing_error &&
      !project.video.is_processing;
    const hasScreenshot = project.media_assets?.some(
      (asset) => asset.kind === 'image'
    );
    const hasLocation =
      project.location && project.location.lat && project.location.lon;
    const hasHomography =
      (project.homography_session as any)?.status === 'solved';

    return hasVideo && hasScreenshot && hasLocation && hasHomography;
  };

  // Helper function to check if video review is available
  const isVideoReviewAvailable = () => {
    return (
      videoUrl &&
      processingRuns?.data?.some((run: any) => run.status === 'completed')
    );
  };

  // Get the latest completed run ID for LLM analysis
  const latestCompletedRun = processingRuns?.data?.find(
    (run: any) => run.status === 'completed'
  );

  const steps = [
    {
      label: 'Setup & Processing',
      description: 'Upload video, configure location, and run analysis',
    },
    {
      label: 'Review Video',
      description: 'Review annotated video with tracking filters',
    },
  ];

  return (
    <Container size='xl' py='xl'>
      <Stack gap='xl'>
        {/* Header */}
        <Group justify='space-between' align='flex-start' wrap='wrap'>
          <Stack gap='xs'>
            <Button
              component={Link}
              to='/projects'
              variant='subtle'
              leftSection={<IconArrowLeft size={16} />}
              size='sm'
            >
              Back to Projects
            </Button>
            <div>
              <Title order={1} mb='xs'>
                {project.title}
              </Title>
              <Text c='dimmed' size='sm'>
                Created on {formatDate(project.created_at)}
              </Text>
            </div>
          </Stack>
          <Group gap='sm'>
            <ActionIcon
              variant='light'
              size='lg'
              onClick={handleEdit}
              title='Edit project'
            >
              <IconEdit size={18} />
            </ActionIcon>
            <ActionIcon
              variant='light'
              color='red'
              size='lg'
              onClick={handleDelete}
              loading={isDeleting}
              title='Delete project'
            >
              <IconTrash size={18} />
            </ActionIcon>
          </Group>
        </Group>

        {/* Main Stepper */}
        <Card withBorder shadow='sm'>
          <Stack gap='md'>
            <Stepper
              active={activeStep}
              onStepClick={setActiveStep}
              allowNextStepsSelect={false}
              size='lg'
            >
              {steps.map((step, index) => (
                <Stepper.Step
                  key={index}
                  label={step.label}
                  description={step.description}
                  icon={
                    index === 0 ? <IconCpu size={18} /> : <IconEye size={18} />
                  }
                  color={
                    index === 0
                      ? isWorkflowComplete()
                        ? 'green'
                        : 'blue'
                      : isVideoReviewAvailable()
                        ? 'green'
                        : 'gray'
                  }
                  completedIcon={<IconCheck size={18} />}
                />
              ))}
            </Stepper>
          </Stack>
        </Card>

        {/* Step Content */}
        {activeStep === 0 ? (
          <ProjectWorkflow
            project={project}
            projectId={params.projectId}
            onRefresh={handleRefresh}
            onReviewVideo={() => setActiveStep(1)}
          />
        ) : (
          <Card withBorder shadow='sm'>
            <Stack gap='md'>
              <Group justify='space-between'>
                <div>
                  <Title order={2} mb='xs'>
                    Video Review & Analysis
                  </Title>
                  <Text size='sm' c='dimmed'>
                    Review the processed video with vehicle tracking annotations
                    and apply filters
                  </Text>
                </div>
                <Button
                  variant='light'
                  onClick={() => setActiveStep(0)}
                  leftSection={<IconArrowLeft size={16} />}
                >
                  Back to Setup
                </Button>
              </Group>

              {videoUrl ? (
                <>
                  {console.log('Video URL:', videoUrl)}
                  {/* Desktop Layout - Side by Side */}
                  <Group
                    align='flex-start'
                    gap='lg'
                    wrap='nowrap'
                    visibleFrom='lg'
                  >
                    {/* Video Annotation - Left Side */}
                    <Box
                      style={{
                        flex: '1 1 60%',
                        minWidth: '400px',
                      }}
                    >
                      <VideoAnnotationViewer
                        videoUrl={videoUrl}
                        projectId={params.projectId}
                        onSeekToTimestamp={handleSeekToTimestamp}
                        videoRef={videoRef}
                      />
                    </Box>

                    {/* AI Accident Analysis - Right Side */}
                    <Box
                      style={{
                        flex: '1 1 40%',
                        minWidth: '350px',
                      }}
                    >
                      <LLMAnalysisPanel
                        projectId={params.projectId}
                        runId={latestCompletedRun?.id}
                        videoRef={videoRef}
                        onSeekToTimestamp={handleSeekToTimestamp}
                      />
                    </Box>
                  </Group>

                  {/* Mobile/Tablet Layout - Stacked */}
                  <Stack gap='lg' hiddenFrom='lg'>
                    <VideoAnnotationViewer
                      videoUrl={videoUrl}
                      projectId={params.projectId}
                      onSeekToTimestamp={handleSeekToTimestamp}
                      videoRef={videoRef}
                    />
                    <LLMAnalysisPanel
                      projectId={params.projectId}
                      runId={latestCompletedRun?.id}
                      videoRef={videoRef}
                      onSeekToTimestamp={handleSeekToTimestamp}
                    />
                  </Stack>

                  {isLoadingVideoUrl && (
                    <Alert
                      color='blue'
                      icon={<IconAlertCircle size={16} />}
                      mt='md'
                    >
                      <Text size='sm'>Loading video URL...</Text>
                    </Alert>
                  )}
                </>
              ) : (
                <Card withBorder p='xl' ta='center'>
                  <Text c='dimmed'>
                    No video available for review. Please complete the setup
                    step first.
                  </Text>
                </Card>
              )}
            </Stack>
          </Card>
        )}
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

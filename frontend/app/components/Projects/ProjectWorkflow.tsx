import { useState, useEffect, useReducer } from 'react';
import {
  Container,
  Stack,
  Group,
  Paper,
  Stepper,
  Button,
  Text,
  Badge,
  Alert,
  Tooltip,
  Grid,
  Transition,
  Box,
  useMantineTheme,
} from '@mantine/core';
import {
  IconVideo,
  IconPhoto,
  IconMapPin,
  IconAdjustments,
  IconCheck,
  IconAlertCircle,
  IconChevronLeft,
  IconChevronRight,
  IconPlayerPlay,
} from '@tabler/icons-react';
import { VideoUpload } from './VideoUpload';
import { LocationAndHomography } from './LocationAndHomography';
import { ProcessingPanel } from '../Processing/ProcessingPanel';
import type { ProjectPublic } from '~/client';

interface ProjectWorkflowProps {
  project: ProjectPublic;
  projectId: string;
  onRefresh: () => void;
  onReviewVideo?: () => void;
}

interface StepStatus {
  completed: boolean;
  warning?: string;
  error?: string;
}

interface WorkflowState {
  activeStep: number;
  stepStatuses: Record<number, StepStatus>;
}

type WorkflowAction =
  | { type: 'SET_ACTIVE_STEP'; payload: number }
  | {
      type: 'UPDATE_STEP_STATUS';
      payload: { step: number; status: StepStatus };
    }
  | { type: 'REFRESH_STATUSES'; payload: ProjectPublic };

function workflowReducer(
  state: WorkflowState,
  action: WorkflowAction
): WorkflowState {
  switch (action.type) {
    case 'SET_ACTIVE_STEP':
      return { ...state, activeStep: action.payload };
    case 'UPDATE_STEP_STATUS':
      return {
        ...state,
        stepStatuses: {
          ...state.stepStatuses,
          [action.payload.step]: action.payload.status,
        },
      };
    case 'REFRESH_STATUSES':
      return {
        ...state,
        stepStatuses: calculateStepStatuses(action.payload),
      };
    default:
      return state;
  }
}

function calculateStepStatuses(
  project: ProjectPublic
): Record<number, StepStatus> {
  const statuses: Record<number, StepStatus> = {};

  // Step 0: Upload Video
  if (project.video) {
    if (project.video.processing_error) {
      statuses[0] = {
        completed: false,
        error: project.video.processing_error,
      };
    } else if (project.video.is_processing) {
      statuses[0] = {
        completed: false,
        warning: 'Video is being processed',
      };
    } else {
      statuses[0] = { completed: true };
    }
  } else {
    statuses[0] = { completed: false, warning: 'No video uploaded' };
  }

  // Step 1: Capture Key Frame
  const hasScreenshot = project.media_assets?.some(
    (asset) => asset.kind === 'image'
  );
  if (hasScreenshot) {
    statuses[1] = { completed: true };
  } else if (project.video?.is_processing) {
    statuses[1] = { completed: false, warning: 'Extracting frame...' };
  } else if (!project.video) {
    statuses[1] = { completed: false, error: 'Upload video first' };
  } else {
    statuses[1] = { completed: false, warning: 'Frame not extracted yet' };
  }

  // Step 2: Set Location & Configure Homography (Combined)
  const homographySession = project.homography_session as any;
  const hasLocation =
    project.location && project.location.lat && project.location.lon;
  const hasHomography = homographySession?.status === 'solved';

  if (hasLocation && hasHomography) {
    statuses[2] = { completed: true };
  } else if (hasLocation && homographySession?.status === 'draft') {
    statuses[2] = {
      completed: false,
      warning: 'Location set, homography configured but not solved',
    };
  } else if (!hasScreenshot) {
    statuses[2] = { completed: false, error: 'Capture key frame first' };
  } else if (!hasLocation) {
    statuses[2] = {
      completed: false,
      warning: 'Location and homography not configured',
    };
  } else {
    statuses[2] = {
      completed: false,
      warning: 'Location set, homography not configured',
    };
  }

  // Step 3: Review & Run
  const allPrerequisitesMet =
    statuses[0]?.completed && statuses[1]?.completed && statuses[2]?.completed;

  statuses[3] = {
    completed: allPrerequisitesMet,
    warning: allPrerequisitesMet ? undefined : 'Complete all prerequisites',
  };

  return statuses;
}

export function ProjectWorkflow({
  project,
  projectId,
  onRefresh,
  onReviewVideo,
}: ProjectWorkflowProps) {
  const theme = useMantineTheme();
  const [state, dispatch] = useReducer(workflowReducer, {
    activeStep: 0,
    stepStatuses: calculateStepStatuses(project),
  });

  const [transitionDirection, setTransitionDirection] = useState<
    'left' | 'right'
  >('right');

  // Recalculate step statuses when project data changes
  useEffect(() => {
    dispatch({ type: 'REFRESH_STATUSES', payload: project });
  }, [project]);

  const handleStepClick = (step: number) => {
    if (step < state.activeStep) {
      setTransitionDirection('left');
    } else {
      setTransitionDirection('right');
    }
    dispatch({ type: 'SET_ACTIVE_STEP', payload: step });
  };

  const handleNext = () => {
    const nextStep = state.activeStep + 1;
    if (nextStep <= 3) {
      setTransitionDirection('right');
      dispatch({ type: 'SET_ACTIVE_STEP', payload: nextStep });
    }
  };

  const handleBack = () => {
    const prevStep = state.activeStep - 1;
    if (prevStep >= 0) {
      setTransitionDirection('left');
      dispatch({ type: 'SET_ACTIVE_STEP', payload: prevStep });
    }
  };

  const canProceedToNext = () => {
    const currentStatus = state.stepStatuses[state.activeStep];
    // Allow proceeding if current step is completed or if it's just a warning
    return currentStatus?.completed || !currentStatus?.error;
  };

  const getStepIcon = (step: number) => {
    const icons = [IconVideo, IconPhoto, IconMapPin, IconPlayerPlay];
    return icons[step];
  };

  const getStepBadge = (step: number) => {
    const status = state.stepStatuses[step];
    if (!status) return null;

    if (status.completed) {
      return (
        <Badge color='green' size='xs' variant='filled'>
          <Group gap={4}>
            <IconCheck size={10} />
            Done
          </Group>
        </Badge>
      );
    }

    if (status.error) {
      return (
        <Tooltip label={status.error}>
          <Badge color='red' size='xs' variant='filled'>
            <Group gap={4}>
              <IconAlertCircle size={10} />
              Error
            </Group>
          </Badge>
        </Tooltip>
      );
    }

    if (status.warning) {
      return (
        <Tooltip label={status.warning}>
          <Badge color='yellow' size='xs' variant='filled'>
            <Group gap={4}>
              <IconAlertCircle size={10} />
              Pending
            </Group>
          </Badge>
        </Tooltip>
      );
    }

    return null;
  };

  const renderStepContent = () => {
    switch (state.activeStep) {
      case 0:
        return (
          <Stack gap='md'>
            <div>
              <Text size='xl' fw={600} mb='xs'>
                Upload Video
              </Text>
              <Text size='sm' c='dimmed'>
                Upload your CCTV or dashcam video for accident analysis. The
                video will be processed to extract a key frame for homography
                configuration.
              </Text>
            </div>

            {project.video && (
              <Alert color='green' icon={<IconCheck size={16} />}>
                Video uploaded successfully
                {project.video.is_processing && ' - Processing frame...'}
              </Alert>
            )}

            {project.video?.processing_error && (
              <Alert color='red' icon={<IconAlertCircle size={16} />}>
                Processing error: {project.video.processing_error}
              </Alert>
            )}

            <VideoUpload projectId={projectId} onUploadComplete={onRefresh} />

            {!project.video && (
              <Alert color='blue' icon={<IconAlertCircle size={16} />}>
                Please upload a video to continue with the workflow.
              </Alert>
            )}
          </Stack>
        );

      case 1:
        return (
          <Stack gap='md'>
            <div>
              <Text size='xl' fw={600} mb='xs'>
                Capture Key Frame
              </Text>
              <Text size='sm' c='dimmed'>
                A key frame from your video is needed for homography
                configuration. This frame will be used to map video coordinates
                to real-world map coordinates.
              </Text>
            </div>

            {(() => {
              const screenshot = project.media_assets?.find(
                (asset) => asset.kind === 'image'
              );

              if (screenshot) {
                return (
                  <Alert color='green' icon={<IconCheck size={16} />}>
                    Key frame captured successfully. You can proceed to set the
                    location.
                  </Alert>
                );
              }

              if (project.video?.is_processing) {
                return (
                  <Alert color='blue' icon={<IconAlertCircle size={16} />}>
                    Frame extraction in progress... This usually takes a few
                    seconds.
                  </Alert>
                );
              }

              if (!project.video) {
                return (
                  <Alert color='red' icon={<IconAlertCircle size={16} />}>
                    Please upload a video first (Step 1).
                  </Alert>
                );
              }

              return (
                <Alert color='yellow' icon={<IconAlertCircle size={16} />}>
                  Frame extraction is pending. The system will automatically
                  extract a frame after video upload. If this takes too long,
                  try refreshing the page.
                </Alert>
              );
            })()}
          </Stack>
        );

      case 2:
        return (
          <Stack gap='md'>
            <div>
              <Text size='xl' fw={600} mb='xs'>
                Set Location & Configure Homography
              </Text>
              <Text size='sm' c='dimmed'>
                Set the geographic location where the video was recorded and
                configure homography mapping to enable accurate speed
                calculations and trajectory mapping.
              </Text>
            </div>

            <LocationAndHomography
              projectId={projectId}
              project={project}
              onLocationSet={onRefresh}
            />
          </Stack>
        );

      case 3:
        return (
          <Stack gap='md'>
            <div>
              <Text size='xl' fw={600} mb='xs'>
                Review & Run Processing
              </Text>
              <Text size='sm' c='dimmed'>
                Review your project configuration and start video processing to
                detect vehicles and calculate speeds.
              </Text>
            </div>

            {/* Prerequisites Summary */}
            <Paper p='sm' withBorder>
              <Text fw={600} mb='md'>
                Prerequisites Summary
              </Text>
              <Stack gap='sm'>
                <Group justify='space-between'>
                  <Group gap='xs'>
                    <IconVideo size={16} />
                    <Text size='sm'>Video Upload</Text>
                  </Group>
                  {state.stepStatuses[0]?.completed ? (
                    <Badge color='green' variant='light'>
                      Complete
                    </Badge>
                  ) : (
                    <Badge color='red' variant='light'>
                      Incomplete
                    </Badge>
                  )}
                </Group>

                <Group justify='space-between'>
                  <Group gap='xs'>
                    <IconPhoto size={16} />
                    <Text size='sm'>Key Frame</Text>
                  </Group>
                  {state.stepStatuses[1]?.completed ? (
                    <Badge color='green' variant='light'>
                      Complete
                    </Badge>
                  ) : (
                    <Badge color='red' variant='light'>
                      Incomplete
                    </Badge>
                  )}
                </Group>

                <Group justify='space-between'>
                  <Group gap='xs'>
                    <IconMapPin size={16} />
                    <Text size='sm'>Location & Homography</Text>
                  </Group>
                  {state.stepStatuses[2]?.completed ? (
                    <Badge color='green' variant='light'>
                      Complete
                    </Badge>
                  ) : (
                    <Badge color='red' variant='light'>
                      Incomplete
                    </Badge>
                  )}
                </Group>
              </Stack>
            </Paper>

            {state.stepStatuses[3]?.completed ? (
              <Alert color='green' icon={<IconCheck size={16} />}>
                All prerequisites met! You can now run video processing.
              </Alert>
            ) : (
              <Alert color='yellow' icon={<IconAlertCircle size={16} />}>
                Please complete all prerequisites before running processing.
                Click on incomplete steps above to complete them.
              </Alert>
            )}

            {/* Processing Panel */}
            <ProcessingPanel
              projectId={projectId}
              homographySolved={
                (project.homography_session as any)?.status === 'solved'
              }
            />
          </Stack>
        );

      default:
        return null;
    }
  };

  const steps = [
    { label: 'Upload Video' },
    { label: 'Capture Frame' },
    {
      label: 'Location & Homography',
    },
    { label: 'Review & Run' },
  ];

  return (
    <Container size='xl' px={{ base: 'xs', sm: 'md' }}>
      <Grid gutter='lg'>
        {/* Left Column: Stepper */}
        <Grid.Col span={{ base: 12, md: 3 }}>
          <Paper p='sm' withBorder style={{ position: 'sticky', top: 20 }}>
            <Stepper
              active={state.activeStep}
              onStepClick={handleStepClick}
              orientation='vertical'
              iconSize={32}
              completedIcon={<IconCheck size={18} />}
            >
              {steps.map((step, index) => {
                const StepIcon = getStepIcon(index);
                return (
                  <Stepper.Step
                    key={index}
                    label={
                      <Group justify='space-between' style={{ width: '100%' }}>
                        <Text size='sm' fw={500}>
                          {step.label}
                        </Text>
                        {getStepBadge(index)}
                      </Group>
                    }
                    icon={<StepIcon size={18} />}
                    color={
                      state.stepStatuses[index]?.completed
                        ? 'green'
                        : state.stepStatuses[index]?.error
                          ? 'red'
                          : state.stepStatuses[index]?.warning
                            ? 'yellow'
                            : 'blue'
                    }
                    allowStepClick
                  />
                );
              })}
            </Stepper>
          </Paper>
        </Grid.Col>

        {/* Right Column: Step Content */}
        <Grid.Col span={{ base: 12, md: 9 }}>
          <Stack gap='md'>
            {/* Content Area with Transition */}
            <Box style={{ position: 'relative', minHeight: 400 }}>
              <Transition
                mounted={true}
                transition={
                  transitionDirection === 'right' ? 'slide-left' : 'slide-right'
                }
                duration={300}
                timingFunction='ease'
              >
                {(styles) => (
                  <Paper withBorder style={styles} p='sm'>
                    {renderStepContent()}
                  </Paper>
                )}
              </Transition>
            </Box>

            {/* Navigation Controls */}
            <Paper p='sm' withBorder>
              <Group justify='space-between'>
                <Button
                  variant='outline'
                  leftSection={<IconChevronLeft size={16} />}
                  onClick={handleBack}
                  disabled={state.activeStep === 0}
                >
                  Back
                </Button>

                <Group gap='xs'>
                  <Text size='sm' c='dimmed'>
                    Step {state.activeStep + 1} of {steps.length}
                  </Text>
                </Group>

                {state.activeStep < 3 ? (
                  <Button
                    rightSection={<IconChevronRight size={16} />}
                    onClick={handleNext}
                    disabled={!canProceedToNext()}
                  >
                    Next
                  </Button>
                ) : (
                  <Button
                    color='green'
                    rightSection={<IconPlayerPlay size={16} />}
                    disabled={!state.stepStatuses[3]?.completed}
                    onClick={() => {
                      if (onReviewVideo) {
                        onReviewVideo();
                      } else {
                        // Scroll to processing panel
                        const processingPanel = document.querySelector(
                          '[data-processing-panel]'
                        );
                        processingPanel?.scrollIntoView({ behavior: 'smooth' });
                      }
                    }}
                  >
                    Review video
                  </Button>
                )}
              </Group>
            </Paper>
          </Stack>
        </Grid.Col>
      </Grid>
    </Container>
  );
}

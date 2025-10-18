import { useState } from 'react';
import {
  Button,
  Card,
  Group,
  Text,
  Badge,
  Stack,
  ActionIcon,
} from '@mantine/core';
import {
  IconCalendar,
  IconMapPin,
  IconEdit,
  IconTrash,
} from '@tabler/icons-react';
import { Link } from 'react-router';
import { useDeleteProject } from '~/hooks/useProjects';
import { useCustomToast } from '~/hooks/useCustomToast';

interface ProjectCardProps {
  project: {
    id: string;
    title: string;
    description?: string;
    created_at: string;
    location?: {
      addr_line?: string;
      lat?: number;
      lon?: number;
    };
    video?: {
      id: string;
      uri: string;
    };
  };
  onEdit?: (project: any) => void;
}

export function ProjectCard({ project, onEdit }: ProjectCardProps) {
  const [isDeleting, setIsDeleting] = useState(false);
  const deleteProject = useDeleteProject();
  const { showToast } = useCustomToast();

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete this project?')) return;

    setIsDeleting(true);
    try {
      await deleteProject.mutateAsync(project.id);
      showToast('Project deleted successfully', 'success');
    } catch (error) {
      showToast('Failed to delete project', 'error');
    } finally {
      setIsDeleting(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  return (
    <Card shadow='sm' padding='lg' radius='md' withBorder>
      <Stack gap='sm'>
        <Group justify='space-between' align='flex-start'>
          <Text fw={500} size='lg' lineClamp={2}>
            {project.title}
          </Text>
          <Group gap='xs'>
            {onEdit && (
              <ActionIcon
                variant='subtle'
                color='blue'
                onClick={() => onEdit(project)}
                size='sm'
              >
                <IconEdit size={16} />
              </ActionIcon>
            )}
            <ActionIcon
              variant='subtle'
              color='red'
              onClick={handleDelete}
              loading={isDeleting}
              size='sm'
            >
              <IconTrash size={16} />
            </ActionIcon>
          </Group>
        </Group>

        {project.description && (
          <Text size='sm' c='dimmed' lineClamp={3}>
            {project.description}
          </Text>
        )}

        <Group gap='xs'>
          <Group gap={4}>
            <IconCalendar size={14} />
            <Text size='xs' c='dimmed'>
              {formatDate(project.created_at)}
            </Text>
          </Group>

          {project.location?.addr_line && (
            <Group gap={4}>
              <IconMapPin size={14} />
              <Text size='xs' c='dimmed'>
                {project.location.addr_line}
              </Text>
            </Group>
          )}
        </Group>

        <Group gap='xs'>
          {project.video && (
            <Badge color='green' variant='light' size='sm'>
              Video uploaded
            </Badge>
          )}
          {project.location && (
            <Badge color='blue' variant='light' size='sm'>
              Location set
            </Badge>
          )}
        </Group>

        <Button
          component={Link}
          to={`/projects/${project.id}`}
          variant='light'
          fullWidth
          mt='md'
        >
          View Details
        </Button>
      </Stack>
    </Card>
  );
}

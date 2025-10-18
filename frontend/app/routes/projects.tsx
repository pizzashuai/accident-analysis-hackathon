import type { Route } from './+types/projects';
import { useState } from 'react';
import {
  Container,
  Title,
  Button,
  Grid,
  Text,
  Stack,
  Group,
  ActionIcon,
} from '@mantine/core';
import { IconPlus, IconRefresh } from '@tabler/icons-react';
import { ProjectCard } from '~/components/Projects/ProjectCard';
import { CreateProjectModal } from '~/components/Projects/CreateProjectModal';
import { useProjects } from '~/hooks/useProjects';
import { useCustomToast } from '~/hooks/useCustomToast';

export function meta({}: Route.MetaArgs) {
  return [
    { title: 'Projects - Accident Analysis' },
    { name: 'description', content: 'Manage your accident analysis projects' },
  ];
}

export default function Projects() {
  const [createModalOpened, setCreateModalOpened] = useState(false);
  const [editingProject, setEditingProject] = useState<any>(null);
  const { data: projectsData, isLoading, error, refetch } = useProjects();
  const { showToast } = useCustomToast();

  const handleRefresh = () => {
    refetch();
    showToast('Projects refreshed', 'success');
  };

  const handleEditProject = (project: any) => {
    setEditingProject(project);
    setCreateModalOpened(true);
  };

  const handleCloseModal = () => {
    setCreateModalOpened(false);
    setEditingProject(null);
  };

  if (isLoading) {
    return (
      <Container>
        <Title order={1} mb='md'>
          Projects
        </Title>
        <Text>Loading projects...</Text>
      </Container>
    );
  }

  if (error) {
    return (
      <Container>
        <Title order={1} mb='md'>
          Projects
        </Title>
        <Text c='red'>Error loading projects. Please try again.</Text>
      </Container>
    );
  }

  const projects = projectsData?.data || [];

  return (
    <Container>
      <Stack gap='lg'>
        <Group justify='space-between' align='center'>
          <div>
            <Title order={1} mb='xs'>
              Projects
            </Title>
            <Text c='dimmed'>Manage your accident analysis projects</Text>
          </div>
          <Group gap='sm'>
            <ActionIcon variant='outline' onClick={handleRefresh} size='lg'>
              <IconRefresh size={20} />
            </ActionIcon>
            <Button
              leftSection={<IconPlus size={16} />}
              onClick={() => setCreateModalOpened(true)}
            >
              Create Project
            </Button>
          </Group>
        </Group>

        {projects.length === 0 ? (
          <Stack align='center' gap='md' py='xl'>
            <Text size='lg' c='dimmed'>
              No projects yet
            </Text>
            <Text c='dimmed'>
              Create your first accident analysis project to get started
            </Text>
            <Button
              leftSection={<IconPlus size={16} />}
              onClick={() => setCreateModalOpened(true)}
            >
              Create Your First Project
            </Button>
          </Stack>
        ) : (
          <Grid>
            {projects.map((project) => (
              <Grid.Col key={project.id} span={{ base: 12, sm: 6, md: 4 }}>
                <ProjectCard
                  project={project as any}
                  onEdit={handleEditProject}
                />
              </Grid.Col>
            ))}
          </Grid>
        )}
      </Stack>

      <CreateProjectModal
        opened={createModalOpened}
        onClose={handleCloseModal}
        editingProject={editingProject}
      />
    </Container>
  );
}

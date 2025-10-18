import { useState } from 'react';
import {
  Modal,
  TextInput,
  Textarea,
  Button,
  Stack,
  Group,
} from '@mantine/core';
import { useForm } from '@mantine/form';
import { useCreateProject, useUpdateProject } from '~/hooks/useProjects';
import { useCustomToast } from '~/hooks/useCustomToast';

interface CreateProjectModalProps {
  opened: boolean;
  onClose: () => void;
  editingProject?: {
    id: string;
    title: string;
    description?: string;
  } | null;
}

export function CreateProjectModal({
  opened,
  onClose,
  editingProject,
}: CreateProjectModalProps) {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const createProject = useCreateProject();
  const updateProject = useUpdateProject();
  const { showToast } = useCustomToast();

  const form = useForm({
    initialValues: {
      title: editingProject?.title || '',
      description: editingProject?.description || '',
    },
    validate: {
      title: (value) => (!value ? 'Title is required' : null),
    },
  });

  const handleSubmit = async (values: typeof form.values) => {
    setIsSubmitting(true);
    try {
      if (editingProject) {
        await updateProject.mutateAsync({
          projectId: editingProject.id,
          data: values,
        });
        showToast('Project updated successfully', 'success');
      } else {
        await createProject.mutateAsync(values);
        showToast('Project created successfully', 'success');
      }
      form.reset();
      onClose();
    } catch (error) {
      showToast(
        editingProject
          ? 'Failed to update project'
          : 'Failed to create project',
        'error'
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    form.reset();
    onClose();
  };

  return (
    <Modal
      opened={opened}
      onClose={handleClose}
      title={editingProject ? 'Edit Project' : 'Create New Project'}
      size='md'
    >
      <form onSubmit={form.onSubmit(handleSubmit)}>
        <Stack gap='md'>
          <TextInput
            label='Project Title'
            placeholder='Enter project title'
            required
            {...form.getInputProps('title')}
          />

          <Textarea
            label='Description'
            placeholder='Enter project description (optional)'
            minRows={3}
            {...form.getInputProps('description')}
          />

          <Group justify='flex-end' gap='sm'>
            <Button variant='outline' onClick={handleClose}>
              Cancel
            </Button>
            <Button type='submit' loading={isSubmitting}>
              {editingProject ? 'Update' : 'Create'} Project
            </Button>
          </Group>
        </Stack>
      </form>
    </Modal>
  );
}

import {
  Box,
  Button,
  Container,
  Flex,
  Group,
  Text,
  TextInput,
  Title,
} from '@mantine/core';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { type SubmitHandler, useForm } from 'react-hook-form';

import {
  type ApiError,
  type UserPublic,
  UsersService,
  type UserUpdateMe,
} from '~/client';
import useAuth from '~/hooks/useAuth';
import useCustomToast from '~/hooks/useCustomToast';
import { emailPattern, handleError } from '~/utils';

const UserInformation = () => {
  const queryClient = useQueryClient();
  const { showSuccessToast } = useCustomToast();
  const [editMode, setEditMode] = useState(false);
  const { user: currentUser } = useAuth();
  const {
    register,
    handleSubmit,
    reset,
    getValues,
    formState: { isSubmitting, errors, isDirty },
  } = useForm<UserPublic>({
    mode: 'onBlur',
    criteriaMode: 'all',
    defaultValues: {
      full_name: currentUser?.full_name,
      email: currentUser?.email,
    },
  });

  const toggleEditMode = () => {
    setEditMode(!editMode);
  };

  const mutation = useMutation({
    mutationFn: (data: UserUpdateMe) =>
      UsersService.updateUserMeRoute({ requestBody: data }),
    onSuccess: () => {
      showSuccessToast('User updated successfully.');
    },
    onError: (err: Error) => {
      handleError(err as ApiError);
    },
    onSettled: () => {
      queryClient.invalidateQueries();
    },
  });

  const onSubmit: SubmitHandler<UserUpdateMe> = async (data) => {
    mutation.mutate(data);
  };

  const onCancel = () => {
    reset();
    toggleEditMode();
  };

  return (
    <Container size='sm'>
      <Title order={3} mb='md'>
        User Information
      </Title>
      <Box component='form' onSubmit={handleSubmit(onSubmit)}>
        <TextInput
          label='Full name'
          {...register('full_name', { maxLength: 30 })}
          disabled={!editMode}
          mb='md'
        />

        <TextInput
          label='Email'
          {...register('email', {
            required: 'Email is required',
            pattern: emailPattern,
          })}
          error={errors.email?.message}
          disabled={!editMode}
          mb='md'
        />

        <Group mt='md'>
          <Button
            type={editMode ? 'submit' : 'button'}
            onClick={editMode ? undefined : toggleEditMode}
            loading={editMode ? isSubmitting : false}
            disabled={editMode ? !isDirty || !getValues('email') : false}
          >
            {editMode ? 'Save' : 'Edit'}
          </Button>
          {editMode && (
            <Button
              variant='subtle'
              color='gray'
              onClick={onCancel}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
          )}
        </Group>
      </Box>
    </Container>
  );
};

export default UserInformation;

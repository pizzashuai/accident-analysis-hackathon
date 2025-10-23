import {
  Box,
  Button,
  Container,
  Stack,
  Title,
  PasswordInput,
  Text,
} from '@mantine/core';
import { useMutation } from '@tanstack/react-query';
import { type SubmitHandler, useForm } from 'react-hook-form';
import { IconLock } from '@tabler/icons-react';

import { type ApiError, type UpdatePassword, UsersService } from '~/client';
import useAuth from '~/hooks/useAuth';
import useCustomToast from '~/hooks/useCustomToast';
import { confirmPasswordRules, handleError, passwordRules } from '~/utils';

interface UpdatePasswordForm extends UpdatePassword {
  confirm_password: string;
}

const ChangePassword = () => {
  const { showSuccessToast } = useCustomToast();
  const { user: currentUser } = useAuth();
  const isDemoUser = currentUser?.email === 'demo@gmail.com';
  const {
    register,
    handleSubmit,
    reset,
    getValues,
    formState: { errors, isSubmitting },
  } = useForm<UpdatePasswordForm>({
    mode: 'onBlur',
    criteriaMode: 'all',
  });

  const mutation = useMutation({
    mutationFn: (data: UpdatePassword) =>
      UsersService.updatePasswordMeRoute({ requestBody: data }),
    onSuccess: () => {
      showSuccessToast('Password updated successfully.');
      reset();
    },
    onError: (err: Error) => {
      handleError(err as ApiError);
    },
  });

  const onSubmit: SubmitHandler<UpdatePasswordForm> = async (data) => {
    mutation.mutate(data);
  };

  return (
    <Container size='sm'>
      <Title order={3} mb='md'>
        Change Password
      </Title>
      <Box component='form' onSubmit={handleSubmit(onSubmit)}>
        <Stack gap='md'>
          <PasswordInput
            label='Current Password'
            leftSection={<IconLock size={16} />}
            {...register('current_password', passwordRules())}
            error={errors.current_password?.message}
            disabled={isDemoUser}
          />
          <PasswordInput
            label='New Password'
            leftSection={<IconLock size={16} />}
            {...register('new_password', passwordRules())}
            error={errors.new_password?.message}
            disabled={isDemoUser}
          />
          <PasswordInput
            label='Confirm Password'
            leftSection={<IconLock size={16} />}
            {...register('confirm_password', confirmPasswordRules(getValues))}
            error={errors.confirm_password?.message}
            disabled={isDemoUser}
          />
        </Stack>
        {isDemoUser && (
          <Text c='dimmed' size='sm' mt='sm'>
            Demo accounts cannot change passwords.
          </Text>
        )}
        <Button
          type='submit'
          loading={isSubmitting}
          mt='md'
          disabled={isDemoUser}
        >
          Save
        </Button>
      </Box>
    </Container>
  );
};
export default ChangePassword;

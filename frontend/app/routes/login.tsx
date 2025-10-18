import {
  Container,
  Image,
  TextInput,
  Text,
  Box,
  Button,
  Anchor,
  Stack,
  PasswordInput,
} from '@mantine/core';
import { Link, useNavigate } from 'react-router';
import type { Route } from './+types/login';
import { type SubmitHandler, useForm } from 'react-hook-form';
import { IconLock, IconMail } from '@tabler/icons-react';
import { useEffect } from 'react';

import type { Body_login_login_access_token as AccessToken } from '~/client';
import useAuth, { isLoggedIn } from '~/hooks/useAuth';
import { emailPattern, passwordRules } from '~/utils';

export default function Login() {
  const navigate = useNavigate();
  const { loginMutation, error, resetError } = useAuth();
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<AccessToken>({
    mode: 'onBlur',
    criteriaMode: 'all',
    defaultValues: {
      username: '',
      password: '',
    },
  });

  useEffect(() => {
    if (isLoggedIn()) {
      navigate('/');
    }
  }, [navigate]);

  const onSubmit: SubmitHandler<AccessToken> = async (data) => {
    if (isSubmitting) return;

    resetError();

    try {
      await loginMutation.mutateAsync(data);
    } catch {
      // error is handled by useAuth hook
    }
  };

  return (
    <Container
      size='xs'
      style={{ minHeight: '100vh', display: 'flex', alignItems: 'center' }}
    >
      <Box
        component='form'
        onSubmit={handleSubmit(onSubmit)}
        style={{ width: '100%' }}
      >
        <Stack gap='md'>
          <Image
            src='https://placehold.co/300x100?text=FastAPI'
            alt='FastAPI logo'
            height={80}
            fit='contain'
            mb='md'
          />

          <TextInput
            label='Email'
            placeholder='Email'
            leftSection={<IconMail size={16} />}
            {...register('username', {
              required: 'Username is required',
              pattern: emailPattern,
            })}
            error={
              errors.username?.message ||
              (error ? 'Invalid credentials' : undefined)
            }
          />

          <PasswordInput
            label='Password'
            placeholder='Password'
            leftSection={<IconLock size={16} />}
            {...register('password', passwordRules())}
            error={errors.password?.message}
          />

          <Anchor component={Link} to='/recover-password' size='sm'>
            Forgot Password?
          </Anchor>

          <Button type='submit' loading={isSubmitting} fullWidth>
            Log In
          </Button>

          <Text ta='center' size='sm'>
            Don't have an account?{' '}
            <Anchor component={Link} to='/signup'>
              Sign Up
            </Anchor>
          </Text>
        </Stack>
      </Box>
    </Container>
  );
}

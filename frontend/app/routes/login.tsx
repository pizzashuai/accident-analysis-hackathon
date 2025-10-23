import {
  Container,
  Flex,
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
    if (isSubmitting || loginMutation.isPending) return;

    resetError();

    try {
      await loginMutation.mutateAsync(data);
    } catch {
      // error is handled by useAuth hook
    }
  };

  const handleDemoLogin = async () => {
    if (loginMutation.isPending) return;

    resetError();

    try {
      await loginMutation.mutateAsync({
        username: 'demo@gmail.com',
        password: '123123123',
      });
    } catch {
      // error is handled by useAuth hook
    }
  };

  const isLoading = isSubmitting || loginMutation.isPending;

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
          <Flex align='center' justify='center' gap='sm' mb='md'>
            <Image
              src='/favicon.png'
              alt='Accident Analysis logo'
              height={80}
              fit='contain'
            />
            <Text size='xl' fw={700}>
              CCTV to Timeline
            </Text>
          </Flex>

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

          <Button type='submit' loading={isLoading} fullWidth>
            Log In
          </Button>

          <Button
            type='button'
            variant='outline'
            onClick={handleDemoLogin}
            loading={isLoading}
            fullWidth
          >
            Log In With Demo Account
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

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
import type { Route } from './+types/signup';
import { type SubmitHandler, useForm } from 'react-hook-form';
import { IconLock, IconUser, IconMail } from '@tabler/icons-react';
import { useEffect } from 'react';

import type { UserRegister } from '~/client';
import useAuth, { isLoggedIn } from '~/hooks/useAuth';
import { confirmPasswordRules, emailPattern, passwordRules } from '~/utils';

interface UserRegisterForm extends UserRegister {
  confirm_password: string;
}

export default function SignUp() {
  const navigate = useNavigate();
  const { signUpMutation } = useAuth();
  const {
    register,
    handleSubmit,
    getValues,
    formState: { errors, isSubmitting },
  } = useForm<UserRegisterForm>({
    mode: 'onBlur',
    criteriaMode: 'all',
    defaultValues: {
      email: '',
      full_name: '',
      password: '',
      confirm_password: '',
    },
  });

  useEffect(() => {
    if (isLoggedIn()) {
      navigate('/');
    }
  }, [navigate]);

  const onSubmit: SubmitHandler<UserRegisterForm> = (data) => {
    signUpMutation.mutate(data);
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
            label='Full Name'
            placeholder='Full Name'
            leftSection={<IconUser size={16} />}
            {...register('full_name', {
              required: 'Full Name is required',
            })}
            error={errors.full_name?.message}
          />

          <TextInput
            label='Email'
            placeholder='Email'
            leftSection={<IconMail size={16} />}
            {...register('email', {
              required: 'Email is required',
              pattern: emailPattern,
            })}
            error={errors.email?.message}
          />

          <PasswordInput
            label='Password'
            placeholder='Password'
            leftSection={<IconLock size={16} />}
            {...register('password', passwordRules())}
            error={errors.password?.message}
          />

          <PasswordInput
            label='Confirm Password'
            placeholder='Confirm Password'
            leftSection={<IconLock size={16} />}
            {...register('confirm_password', confirmPasswordRules(getValues))}
            error={errors.confirm_password?.message}
          />

          <Button type='submit' loading={isSubmitting} fullWidth>
            Sign Up
          </Button>

          <Text ta='center' size='sm'>
            Already have an account?{' '}
            <Anchor component={Link} to='/login'>
              Log In
            </Anchor>
          </Text>
        </Stack>
      </Box>
    </Container>
  );
}

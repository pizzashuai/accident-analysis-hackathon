import type { Route } from './+types/home';
import { Welcome } from '../welcome/welcome';
import { Container, Text, Title } from '@mantine/core';
import useAuth from '~/hooks/useAuth';

export function meta({}: Route.MetaArgs) {
  return [
    { title: 'Home - Protected Route' },
    { name: 'description', content: 'Welcome to the protected home page!' },
  ];
}

export default function Home() {
  const { user } = useAuth();

  return (
    <Container>
      <Title order={1} mb='md'>
        Welcome, {user?.full_name || user?.email || 'User'}!
      </Title>
      <Text mb='xl'>
        This is a protected route. You can only see this page when logged in.
      </Text>
      <Welcome />
    </Container>
  );
}

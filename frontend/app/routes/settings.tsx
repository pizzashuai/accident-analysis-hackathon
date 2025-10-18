import { Container, Tabs, Title } from '@mantine/core';
import type { Route } from './+types/settings';
import UserInformation from '~/components/UserSettings/UserInformation';
import ChangePassword from '~/components/UserSettings/ChangePassword';

export function meta({}: Route.MetaArgs) {
  return [
    { title: 'User Settings' },
    { name: 'description', content: 'Manage your account settings' },
  ];
}

export default function Settings() {
  return (
    <Container size='lg' py='xl'>
      <Title order={2} mb='xl'>
        User Settings
      </Title>

      <Tabs defaultValue='profile'>
        <Tabs.List>
          <Tabs.Tab value='profile'>Profile</Tabs.Tab>
          <Tabs.Tab value='password'>Password</Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value='profile' pt='md'>
          <UserInformation />
        </Tabs.Panel>

        <Tabs.Panel value='password' pt='md'>
          <ChangePassword />
        </Tabs.Panel>
      </Tabs>
    </Container>
  );
}

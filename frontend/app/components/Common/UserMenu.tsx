import { Button, Menu, Text, rem } from '@mantine/core';
import { Link } from 'react-router';
import { IconUserCircle, IconLogout, IconSettings } from '@tabler/icons-react';

import useAuth from '~/hooks/useAuth';

const UserMenu = () => {
  const { user, logout } = useAuth();

  const handleLogout = async () => {
    logout();
  };

  return (
    <Menu shadow='md' width={200}>
      <Menu.Target>
        <Button
          leftSection={<IconUserCircle size={18} />}
          variant='light'
          data-testid='user-menu'
        >
          {user?.full_name || 'User'}
        </Button>
      </Menu.Target>

      <Menu.Dropdown>
        <Menu.Label>Account</Menu.Label>
        <Menu.Item
          component={Link}
          to='/settings'
          leftSection={
            <IconSettings style={{ width: rem(14), height: rem(14) }} />
          }
        >
          My Profile
        </Menu.Item>

        <Menu.Divider />

        <Menu.Item
          color='red'
          leftSection={
            <IconLogout style={{ width: rem(14), height: rem(14) }} />
          }
          onClick={handleLogout}
        >
          Log Out
        </Menu.Item>
      </Menu.Dropdown>
    </Menu>
  );
};

export default UserMenu;

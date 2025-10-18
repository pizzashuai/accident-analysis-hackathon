import { Flex, Image, Group, NavLink } from '@mantine/core';
import { Link, useLocation } from 'react-router';

import UserMenu from './UserMenu';

function Navbar() {
  const location = useLocation();

  return (
    <Flex
      justify='space-between'
      align='center'
      style={(theme) => ({
        backgroundColor: theme.colors.gray[1],
        borderBottom: `1px solid ${theme.colors.gray[3]}`,
        position: 'sticky',
        top: 0,
        zIndex: 100,
      })}
      p='md'
    >
      <Group gap='lg'>
        <Link to='/' style={{ textDecoration: 'none' }}>
          <Image
            src='/logo.png'
            alt='Logo'
            h={40}
            w='auto'
            fallbackSrc='https://placehold.co/150x40?text=Logo'
          />
        </Link>

        <Group gap='xs'>
          <NavLink
            component={Link}
            to='/'
            label='Home'
            active={location.pathname === '/'}
            style={{ textDecoration: 'none' }}
          />
          <NavLink
            component={Link}
            to='/projects'
            label='Projects'
            active={location.pathname.startsWith('/projects')}
            style={{ textDecoration: 'none' }}
          />
        </Group>
      </Group>

      <Group gap='sm'>
        <UserMenu />
      </Group>
    </Flex>
  );
}

export default Navbar;

import { Flex, Group, Title } from '@mantine/core';

import UserMenu from './UserMenu';

function Navbar() {
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
        <Title order={2} c='blue'>
          Shuai Accident Analysis
        </Title>
      </Group>

      <Group gap='sm'>
        <UserMenu />
      </Group>
    </Flex>
  );
}

export default Navbar;

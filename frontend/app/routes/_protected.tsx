import { Flex, Box } from '@mantine/core';
import { Outlet, useNavigate } from 'react-router';
import { useEffect } from 'react';

import Navbar from '~/components/Common/Navbar';
import { isLoggedIn } from '~/hooks/useAuth';

export default function ProtectedLayout() {
  const navigate = useNavigate();

  useEffect(() => {
    if (!isLoggedIn()) {
      navigate('/login');
    }
  }, [navigate]);

  return (
    <Flex direction='column' h='100vh'>
      <Navbar />
      <Box style={{ flex: 1, overflowY: 'auto' }} p='md'>
        <Outlet />
      </Box>
    </Flex>
  );
}

'use client';

import { notifications } from '@mantine/notifications';

const useCustomToast = () => {
  const showToast = (
    message: string,
    type: 'success' | 'error' | 'info' = 'info'
  ) => {
    notifications.show({
      title:
        type === 'success'
          ? 'Success!'
          : type === 'error'
            ? 'Something went wrong!'
            : 'Info',
      message,
      color: type === 'success' ? 'green' : type === 'error' ? 'red' : 'blue',
    });
  };

  return { showToast };
};

export default useCustomToast;
export { useCustomToast };

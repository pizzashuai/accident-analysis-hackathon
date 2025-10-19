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

  const showSuccessToast = (message: string) => {
    showToast(message, 'success');
  };

  const showErrorToast = (message: string) => {
    showToast(message, 'error');
  };

  const showInfoToast = (message: string) => {
    showToast(message, 'info');
  };

  return { showToast, showSuccessToast, showErrorToast, showInfoToast };
};

export default useCustomToast;
export { useCustomToast };

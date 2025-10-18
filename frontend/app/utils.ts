import type { ApiError } from '~/client';
import { notifications } from '@mantine/notifications';

export const emailPattern = {
  value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
  message: 'Invalid email address',
};

export const passwordRules = (isRequired = true) => {
  return {
    required: isRequired ? 'Password is required' : false,
    minLength: {
      value: 8,
      message: 'Password must be at least 8 characters',
    },
    maxLength: {
      value: 40,
      message: 'Password must be at most 40 characters',
    },
  };
};

export const confirmPasswordRules = (getValues: () => any) => ({
  required: 'Please confirm your password',
  validate: (value: string) =>
    value === getValues().password || 'The passwords do not match',
});

export const handleError = (err: ApiError) => {
  let errDetail = (err.body as any)?.detail;

  let errorMessage: string;

  if (Array.isArray(errDetail) && errDetail.length > 0) {
    errorMessage = errDetail.map((detail: any) => detail.msg).join('; ');
  } else if (typeof errDetail === 'string') {
    errorMessage = errDetail;
  } else {
    errorMessage = 'Something went wrong. Please try again later.';
  }

  notifications.show({
    title: 'Something went wrong!',
    message: errorMessage,
    color: 'red',
  });
};

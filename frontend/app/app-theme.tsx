import {
  createTheme,
  DEFAULT_THEME,
  MantineProvider,
  type MantineProviderProps,
} from '@mantine/core';
import { DatesProvider } from '@mantine/dates';
import 'dayjs/locale/en';

export const appTheme = createTheme({
  colors: {
    brand: DEFAULT_THEME.colors.blue,
  },
  primaryColor: 'brand',
});

export function AppTheme({
  children,
  theme = appTheme,
  ...props
}: MantineProviderProps) {
  return (
    <MantineProvider theme={theme} withNormalizeCSS withGlobalStyles {...props}>
      <DatesProvider settings={{ locale: 'en' }}>{children}</DatesProvider>
    </MantineProvider>
  );
}

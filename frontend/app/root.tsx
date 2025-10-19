import React from 'react';
import {
  isRouteErrorResponse,
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
} from 'react-router';
import {
  Box,
  Code,
  ColorSchemeScript,
  Container,
  mantineHtmlProps,
  Text,
  Title,
} from '@mantine/core';
import { Notifications } from '@mantine/notifications';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Route } from './+types/root';
import './app.css';
import '@mantine/notifications/styles.css';
import { AppTheme } from '~/app-theme';
import { OpenAPI } from '~/client';

// Configure OpenAPI client
OpenAPI.BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
OpenAPI.TOKEN = async () => {
  return localStorage.getItem('access_token') || '';
};

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      refetchOnWindowFocus: false,
    },
  },
});

export const links: Route.LinksFunction = () => [
  { rel: 'icon', href: '/favicon.ico', type: 'image/x-icon' },
  { rel: 'icon', href: '/favicon.png', type: 'image/png' },
];

export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang='en' {...mantineHtmlProps}>
      <head>
        <meta charSet='utf-8' />
        <meta
          name='viewport'
          content='width=device-width, initial-scale=1, maximum-scale=1'
        />
        <ColorSchemeScript />
        <Meta />
        <Links />
      </head>
      <body>
        <QueryClientProvider client={queryClient}>
          <AppTheme>
            {children}
            <Notifications position='top-right' />
          </AppTheme>
        </QueryClientProvider>
        <ScrollRestoration />
        <Scripts />
      </body>
    </html>
  );
}

export default function App() {
  return <Outlet />;
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  let message = 'Oops!';
  let details = 'An unexpected error occurred.';
  let stack: string | undefined;

  if (isRouteErrorResponse(error)) {
    message = error.status === 404 ? '404' : 'Error';
    details =
      error.status === 404
        ? 'The requested page could not be found.'
        : error.statusText || details;
  } else if (import.meta.env.DEV && error && error instanceof Error) {
    details = error.message;
    stack = error.stack;
  }

  return (
    <AppTheme>
      <Container component='main' pt='xl' p='md' mx='auto'>
        <Title>{message}</Title>
        <Text>{details}</Text>
        {stack && (
          <Box component='pre' w='100%' style={{ overflowX: 'auto' }} p='md'>
            <Code>{stack}</Code>
          </Box>
        )}
      </Container>
    </AppTheme>
  );
}

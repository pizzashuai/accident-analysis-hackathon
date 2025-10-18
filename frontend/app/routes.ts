import {
  type RouteConfig,
  index,
  layout,
  route,
} from '@react-router/dev/routes';

export default [
  // Public routes
  route('login', 'routes/login.tsx'),
  route('signup', 'routes/signup.tsx'),

  // Protected routes
  layout('routes/_protected.tsx', [
    index('routes/home.tsx'),
    route('projects', 'routes/projects.tsx'),
    route('projects/:projectId', 'routes/projects.$projectId.tsx'),
    route('settings', 'routes/settings.tsx'),
  ]),
] satisfies RouteConfig;

# Authentication System Guide

## Quick Start

The application now has a complete authentication system with protected routes. Here's what you need to know:

### Starting the Application

```bash
cd frontend
pnpm install  # If you haven't already
pnpm dev
```

Visit `http://localhost:5173` - you'll be automatically redirected to `/login` if not authenticated.

## File Structure

```
app/
├── hooks/
│   ├── useAuth.ts              # Authentication hook (login, signup, logout, user state)
│   └── useCustomToast.ts       # Toast notifications
│
├── components/
│   ├── Common/
│   │   ├── Navbar.tsx          # Top navigation bar with user menu
│   │   └── UserMenu.tsx        # User dropdown (profile, logout)
│   │
│   └── UserSettings/
│       ├── UserInformation.tsx # Edit profile form
│       └── ChangePassword.tsx  # Change password form
│
├── routes/
│   ├── login.tsx               # Login page (public)
│   ├── signup.tsx              # Signup page (public)
│   ├── settings.tsx            # User settings (protected)
│   ├── _protected.tsx          # Protected route layout wrapper
│   └── home.tsx                # Home page (protected)
│
├── utils.ts                    # Validation rules and error handling
├── root.tsx                    # App root with QueryClient and Notifications
└── routes.ts                   # Route configuration
```

## Route Configuration

### Public Routes

- `/login` - Login page (redirects to `/` if already logged in)
- `/signup` - Signup page (redirects to `/` if already logged in)

### Protected Routes (requires authentication)

All routes under the `_protected` layout require authentication:

- `/` - Home page with welcome message
- `/settings` - User settings with tabs for Profile and Password

## Key Features

### 1. Authentication Hook (`useAuth`)

```typescript
import useAuth from '~/hooks/useAuth';

function MyComponent() {
  const { user, loginMutation, signUpMutation, logout } = useAuth();

  // user: Current user object (null if not logged in)
  // loginMutation: Login mutation from react-query
  // signUpMutation: Signup mutation from react-query
  // logout: Logout function
}
```

### 2. Protected Routes

The `_protected.tsx` layout automatically:

- Checks if user is logged in
- Redirects to `/login` if not authenticated
- Shows Navbar with user menu
- Wraps all protected pages

### 3. Toast Notifications

```typescript
import useCustomToast from '~/hooks/useCustomToast';

function MyComponent() {
  const { showSuccessToast, showErrorToast } = useCustomToast();

  showSuccessToast('Operation successful!');
  showErrorToast('Something went wrong!');
}
```

### 4. Form Validation

All forms use `react-hook-form` with built-in validation:

- Email: Valid email format required
- Password: 8-40 characters
- Confirm Password: Must match password field

## User Flow

### First Time User

1. Visit app → Redirected to `/login`
2. Click "Sign Up" link
3. Fill in registration form (full name, email, password)
4. Submit → Automatically logged in → Redirected to home page
5. See personalized welcome message
6. Access user menu in top right (click user icon)
7. Navigate to settings to update profile

### Returning User

1. Visit app → Redirected to `/login`
2. Enter email and password
3. Submit → Redirected to home page
4. Access all protected routes

### Logged In User

- Top navigation bar always visible with:
  - Logo (clickable, goes to home)
  - User menu button (shows name)
    - "My Profile" → Goes to `/settings`
    - "Log Out" → Logs out and redirects to `/login`

## API Integration

### Automatic Token Management

The OpenAPI client is configured in `root.tsx`:

```typescript
OpenAPI.BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
OpenAPI.TOKEN = async () => localStorage.getItem('access_token') || '';
```

All API requests automatically include the JWT token in headers.

### Environment Variables

Create a `.env` file in the frontend directory:

```
VITE_API_URL=http://localhost:8000
```

## Styling

### Mantine Theme

The app uses Mantine UI v8 with a custom theme defined in `app-theme.tsx`:

- Primary color: Blue (brand)
- Responsive design
- Dark mode ready (if you enable it)

### Custom Styling

All components use Mantine's inline styles and props:

```typescript
<Box p="md" bg="gray.1">  // Padding medium, background gray.1
<Button variant="filled" color="blue">  // Filled button, blue color
<Title order={1}>  // H1 heading
```

## Adding New Protected Routes

1. Create your route file in `app/routes/`:

```typescript
// app/routes/my-page.tsx
import { Container, Title } from "@mantine/core";

export default function MyPage() {
  return (
    <Container>
      <Title>My Protected Page</Title>
    </Container>
  );
}
```

2. Add to `routes.ts` inside the protected layout:

```typescript
layout("routes/_protected.tsx", [
  index("routes/home.tsx"),
  route("settings", "routes/settings.tsx"),
  route("my-page", "routes/my-page.tsx"),  // Add this line
]),
```

That's it! The route is now protected and will show the Navbar.

## Security Notes

- JWT tokens are stored in localStorage
- Tokens are automatically included in all API requests
- Protected routes check authentication on every navigation
- Logout clears the token and redirects to login
- Failed API requests (401/403) should be handled by the error boundary

## Troubleshooting

### "Cannot read property 'access_token' of null"

- The backend API is not running
- Check `VITE_API_URL` environment variable

### Infinite redirect loop

- Token might be invalid
- Clear localStorage: `localStorage.removeItem('access_token')`

### Styles not loading

- Ensure `@mantine/core/styles.css` is imported in `app.css`
- Check that `@mantine/notifications/styles.css` is imported in `root.tsx`

### TypeScript errors in old_src

- The `old_src` folder uses Chakra UI and is deprecated
- You can safely ignore or delete it after confirming the new implementation works

## Next Steps

You can now:

1. Test the authentication flow
2. Add more protected routes
3. Customize the Navbar
4. Add profile pictures
5. Implement password recovery
6. Add email verification
7. Enhance the user settings page

Enjoy your new authentication system! 🚀

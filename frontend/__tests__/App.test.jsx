import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import App from '../App';
import { useAuthStore } from '../store/authStore';

// Mock all route components
jest.mock('../pages/SignUpPage', () => {
  return function MockSignUpPage() {
    return <div data-testid="signup-page">SignUp Page</div>;
  };
});

jest.mock('../pages/LoginPage', () => {
  return function MockLoginPage() {
    return <div data-testid="login-page">Login Page</div>;
  };
});

jest.mock('../pages/EmailVerificationPage', () => {
  return function MockEmailVerificationPage() {
    return <div data-testid="email-verification-page">Email Verification Page</div>;
  };
});

jest.mock('../pages/DashboardPage', () => {
  return function MockDashboardPage() {
    return <div data-testid="dashboard-page">Dashboard Page</div>;
  };
});

jest.mock('../pages/ForgotPasswordPage', () => {
  return function MockForgotPasswordPage() {
    return <div data-testid="forgot-password-page">Forgot Password Page</div>;
  };
});

jest.mock('../pages/ResetPasswordPage', () => {
  return function MockResetPasswordPage() {
    return <div data-testid="reset-password-page">Reset Password Page</div>;
  };
});

jest.mock('../pages/LandingPage', () => {
  return function MockLandingPage() {
    return <div data-testid="landing-page">Landing Page</div>;
  };
});

jest.mock('../pages/WatchlistPage', () => {
  return function MockWatchlistPage() {
    return <div data-testid="watchlist-page">Watchlist Page</div>;
  };
});

jest.mock('../pages/AboutPage', () => {
  return function MockAboutPage() {
    return <div data-testid="about-page">About Page</div>;
  };
});

jest.mock('../pages/AccountPage', () => {
  return function MockAccountPage() {
    return <div data-testid="account-page">Account Page</div>;
  };
});

jest.mock('../pages/SearchResultsPage', () => {
  return function MockSearchResultsPage() {
    return <div data-testid="search-results-page">Search Results Page</div>;
  };
});

jest.mock('../pages/OnboardingPage', () => {
  return function MockOnboardingPage() {
    return <div data-testid="onboarding-page">Onboarding Page</div>;
  };
});

jest.mock('../components/LoadingSpinner', () => {
  return function MockLoadingSpinner() {
    return <div data-testid="loading-spinner">Loading Spinner</div>;
  };
});

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }) => <div {...props}>{children}</div>
  },
  AnimatePresence: ({ children }) => <>{children}</>,
}));

describe('App Component', () => {
  beforeEach(() => {
    // Act
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );
    
    // Assert
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });

  it('renders landing page for unauthenticated user', async () => {
    // Arrange
    useAuthStore.setState({
      isCheckingAuth: false,
      isAuthenticated: false,
      user: null
    });
    
    // Act
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );
    
    // Assert
    expect(screen.getByTestId('landing-page')).toBeInTheDocument();
  });

  it('renders dashboard page for authenticated user', async () => {
    // Arrange
    useAuthStore.setState({
      isCheckingAuth: false,
      isAuthenticated: true,
      user: { id: '1', name: 'Test User', isVerified: true, hasOnboarded: true }
    });
    
    // Navigate to default route
    window.history.pushState({}, '', '/');
    
    // Act
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );
    
    // Assert
    expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
  });

  it('renders onboarding page for authenticated user who has not onboarded', async () => {
    // Arrange
    useAuthStore.setState({
      isCheckingAuth: false,
      isAuthenticated: true,
      user: { id: '1', name: 'Test User', isVerified: true, hasOnboarded: false }
    });
    
    // Navigate to default route
    window.history.pushState({}, '', '/');
    
    // Act
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );
    
    // Assert
    expect(screen.getByTestId('onboarding-page')).toBeInTheDocument();
  });

  it('redirects to login for protected routes when user is not authenticated', async () => {
    // Arrange
    useAuthStore.setState({
      isCheckingAuth: false,
      isAuthenticated: false,
      user: null
    });
    
    // Navigate to a protected route
    window.history.pushState({}, '', '/dashboard');
    
    // Act
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );
    
    // Assert
    expect(screen.getByTestId('login-page')).toBeInTheDocument();
  });

  it('renders signup page when navigating to /signup', async () => {
    // Arrange
    useAuthStore.setState({
      isCheckingAuth: false,
      isAuthenticated: false,
      user: null
    });
    
    // Navigate to signup route
    window.history.pushState({}, '', '/signup');
    
    // Act
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );
    
    // Assert
    expect(screen.getByTestId('signup-page')).toBeInTheDocument();
  });

  it('renders login page when navigating to /login', async () => {
    // Arrange
    useAuthStore.setState({
      isCheckingAuth: false,
      isAuthenticated: false,
      user: null
    });
    
    // Navigate to login route
    window.history.pushState({}, '', '/login');
    
    // Act
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );
    
    // Assert
    expect(screen.getByTestId('login-page')).toBeInTheDocument();
  });

  it('renders email verification page when navigating to /verify-email', async () => {
    // Arrange
    useAuthStore.setState({
      isCheckingAuth: false,
      isAuthenticated: false,
      user: null
    });
    
    // Navigate to verify email route
    window.history.pushState({}, '', '/verify-email');
    
    // Act
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );
    
    // Assert
    expect(screen.getByTestId('email-verification-page')).toBeInTheDocument();
  });

  it('renders forgot password page when navigating to /forgot-password', async () => {
    // Arrange
    useAuthStore.setState({
      isCheckingAuth: false,
      isAuthenticated: false,
      user: null
    });
    
    // Navigate to forgot password route
    window.history.pushState({}, '', '/forgot-password');
    
    // Act
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );
    
    // Assert
    expect(screen.getByTestId('forgot-password-page')).toBeInTheDocument();
  });

  it('renders reset password page when navigating to /reset-password/:token', async () => {
    // Arrange
    useAuthStore.setState({
      isCheckingAuth: false,
      isAuthenticated: false,
      user: null
    });
    
    // Navigate to reset password route
    window.history.pushState({}, '', '/reset-password/some-token');
    
    // Act
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );
    
    // Assert
    expect(screen.getByTestId('reset-password-page')).toBeInTheDocument();
  });

  it('redirects to dashboard for authenticated user trying to access auth pages', async () => {
    // Arrange
    useAuthStore.setState({
      isCheckingAuth: false,
      isAuthenticated: true,
      user: { id: '1', name: 'Test User', isVerified: true, hasOnboarded: true }
    });
    
    // Navigate to login route
    window.history.pushState({}, '', '/login');
    
    // Act
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );
    
    // Assert
    expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
  });

  it('calls checkAuth on mount', async () => {
    // Arrange
    const checkAuthMock = jest.fn();
    useAuthStore.setState({
      isCheckingAuth: false,
      isAuthenticated: false,
      user: null,
      checkAuth: checkAuthMock
    });
    
    // Act
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );
    
    // Assert
    expect(checkAuthMock).toHaveBeenCalledTimes(1);
  });
}); Reset history and location
    window.history.pushState({}, '', '/');
    
    // Reset auth store before each test
    useAuthStore.setState({
      isCheckingAuth: false,
      isAuthenticated: false,
      user: null,
      checkAuth: jest.fn()
    });
  });

  it('renders loading spinner while checking authentication', async () => {
    // Arrange
    useAuthStore.setState({
      isCheckingAuth: true
    });
    
    //
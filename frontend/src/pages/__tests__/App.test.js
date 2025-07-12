import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import App from '../../App';
import { useAuthStore } from '../../store/authStore';


jest.mock('../../store/authStore', () => ({
  useAuthStore: jest.fn()
}));


jest.mock('react-hot-toast', () => ({
  Toaster: () => <div data-testid="toaster-mock" />
}));


jest.mock('../LandingPage', () => () => <div data-testid="landing-page-mock">Landing Page</div>);
jest.mock('../SignUpPage', () => () => <div data-testid="signup-page-mock">Sign Up Page</div>);
jest.mock('../LoginPage', () => () => <div data-testid="login-page-mock">Login Page</div>);
jest.mock('../DashboardPage', () => () => <div data-testid="dashboard-page-mock">Dashboard Page</div>);
jest.mock('../OnboardingPage', () => () => <div data-testid="onboarding-page-mock">Onboarding Page</div>);
jest.mock('../WatchlistPage', () => () => <div data-testid="watchlist-page-mock">Watchlist Page</div>);
jest.mock('../AboutPage', () => () => <div data-testid="about-page-mock">About Page</div>);
jest.mock('../AccountPage', () => () => <div data-testid="account-page-mock">Account Page</div>);
jest.mock('../SearchResultsPage', () => () => <div data-testid="search-results-page-mock">Search Results Page</div>);
jest.mock('../ForgotPasswordPage', () => () => <div data-testid="forgot-password-page-mock">Forgot Password Page</div>);


jest.mock('../../components/LoadingSpinner', () => () => <div data-testid="loading-spinner-mock">Loading...</div>);


jest.mock('../../components/FloatingShape', () => () => <div data-testid="floating-shape-mock" />);

describe('App Component', () => {

  const defaultAuthStore = {
    isCheckingAuth: false,
    isAuthenticated: false,
    user: null,
    checkAuth: jest.fn()
  };

  beforeEach(() => {

    useAuthStore.mockReturnValue(defaultAuthStore);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('shows loading spinner when checking auth', () => {
    useAuthStore.mockReturnValue({
      ...defaultAuthStore,
      isCheckingAuth: true
    });
    
    render(<App />);
    
    expect(screen.getByTestId('loading-spinner-mock')).toBeInTheDocument();
  });

  test('renders landing page for unauthenticated users', () => {
    render(
      <MemoryRouter initialEntries={['/']}>
        <App />
      </MemoryRouter>
    );
    
    expect(screen.getByTestId('landing-page-mock')).toBeInTheDocument();
  });

  test('renders sign up page for unauthenticated users', () => {
    render(
      <MemoryRouter initialEntries={['/signup']}>
        <App />
      </MemoryRouter>
    );
    
    expect(screen.getByTestId('signup-page-mock')).toBeInTheDocument();
  });

  test('renders login page for unauthenticated users', () => {
    render(
      <MemoryRouter initialEntries={['/login']}>
        <App />
      </MemoryRouter>
    );
    
    expect(screen.getByTestId('login-page-mock')).toBeInTheDocument();
  });

  test('redirects authenticated users to dashboard from landing page', () => {
    useAuthStore.mockReturnValue({
      ...defaultAuthStore,
      isAuthenticated: true,
      user: { hasOnboarded: true }
    });
    
    render(
      <MemoryRouter initialEntries={['/']}>
        <App />
      </MemoryRouter>
    );
    
    expect(screen.getByTestId('dashboard-page-mock')).toBeInTheDocument();
    expect(screen.queryByTestId('landing-page-mock')).not.toBeInTheDocument();
  });

  test('redirects authenticated users to onboarding if not onboarded yet', () => {
    useAuthStore.mockReturnValue({
      ...defaultAuthStore,
      isAuthenticated: true,
      user: { hasOnboarded: false }
    });
    
    render(
      <MemoryRouter initialEntries={['/login']}>
        <App />
      </MemoryRouter>
    );
    
    expect(screen.getByTestId('onboarding-page-mock')).toBeInTheDocument();
    expect(screen.queryByTestId('login-page-mock')).not.toBeInTheDocument();
  });

  test('redirects unauthenticated users to login from protected routes', () => {
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <App />
      </MemoryRouter>
    );
    
    expect(screen.getByTestId('login-page-mock')).toBeInTheDocument();
    expect(screen.queryByTestId('dashboard-page-mock')).not.toBeInTheDocument();
  });

  test('renders dashboard for authenticated and onboarded users', () => {
    useAuthStore.mockReturnValue({
      ...defaultAuthStore,
      isAuthenticated: true,
      user: { hasOnboarded: true }
    });
    
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <App />
      </MemoryRouter>
    );
    
    expect(screen.getByTestId('dashboard-page-mock')).toBeInTheDocument();
  });

  test('renders watchlist for authenticated users', () => {
    useAuthStore.mockReturnValue({
      ...defaultAuthStore,
      isAuthenticated: true,
      user: { hasOnboarded: true }
    });
    
    render(
      <MemoryRouter initialEntries={['/watchlist']}>
        <App />
      </MemoryRouter>
    );
    
    expect(screen.getByTestId('watchlist-page-mock')).toBeInTheDocument();
  });

  test('renders search results page', () => {
    render(
      <MemoryRouter initialEntries={['/search/avengers']}>
        <App />
      </MemoryRouter>
    );
    
    expect(screen.getByTestId('search-results-page-mock')).toBeInTheDocument();
  });

  test('redirects to landing page for unknown routes', () => {
    render(
      <MemoryRouter initialEntries={['/unknown-route']}>
        <App />
      </MemoryRouter>
    );
    
    expect(screen.getByTestId('landing-page-mock')).toBeInTheDocument();
  });

  test('calls checkAuth on mount', () => {
    render(
      <MemoryRouter initialEntries={['/']}>
        <App />
      </MemoryRouter>
    );
    
    expect(defaultAuthStore.checkAuth).toHaveBeenCalled();
  });
});
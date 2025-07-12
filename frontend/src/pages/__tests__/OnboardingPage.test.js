import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import OnboardingPage from '../OnboardingPage';
import { useAuthStore } from '../../store/authStore';
import { useNavigate } from 'react-router-dom';

jest.mock('../../store/authStore', () => ({
  useAuthStore: jest.fn()
}));

jest.mock('react-router-dom', () => ({
  useNavigate: jest.fn()
}));

jest.mock('../../utils/tmdbIds', () => ({
  VALID_TMDB_IDS: new Set(['123', '456', '789'])
}));

describe('OnboardingPage Component', () => {
  const mockSearchResults = {
    results: [
      {
        id: 123,
        title: 'Test Movie 1',
        poster_path: '/poster1.jpg',
        release_date: '2023-01-01'
      },
      {
        id: 456,
        title: 'Test Movie 2',
        poster_path: '/poster2.jpg',
        release_date: '2023-02-01'
      }
    ]
  };

  const mockStoreData = {
    user: { 
      _id: 'user123', 
      name: 'Test User',
      hasOnboarded: false
    },
    isAuthenticated: true,
    updateOnboarding: jest.fn().mockResolvedValue({}),
    logout: jest.fn()
  };

  const mockNavigate = jest.fn();

  beforeEach(() => {
    useAuthStore.mockReturnValue(mockStoreData);
    useNavigate.mockReturnValue(mockNavigate);
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockSearchResults)
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('renders onboarding page with search form', () => {
    render(<OnboardingPage />);
    expect(screen.getByText('What movies have you loved?')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Search for a movie...')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Search' })).toBeInTheDocument();
    const finishButton = screen.getByRole('button', { name: 'Finish Onboarding' });
    expect(finishButton).toBeInTheDocument();
    expect(finishButton).toBeDisabled();
  });

  test('redirects to dashboard if user has already onboarded', () => {
    useAuthStore.mockReturnValue({
      ...mockStoreData,
      user: { ...mockStoreData.user, hasOnboarded: true }
    });
    render(<OnboardingPage />);
    expect(mockNavigate).toHaveBeenCalledWith('/dashboard');
  });

  test('handles search functionality', async () => {
    const user = userEvent.setup();
    render(<OnboardingPage />);
    const searchInput = screen.getByPlaceholderText('Search for a movie...');
    await user.type(searchInput, 'avengers');
    const searchButton = screen.getByRole('button', { name: 'Search' });
    await user.click(searchButton);
    expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining('query=avengers'));
    await waitFor(() => {
      expect(screen.getByText('Test Movie 1')).toBeInTheDocument();
      expect(screen.getByText('Test Movie 2')).toBeInTheDocument();
    });
  });

  test('allows adding movies to favorites', async () => {
    render(<OnboardingPage />);
    const searchInput = screen.getByPlaceholderText('Search for a movie...');
    fireEvent.change(searchInput, { target: { value: 'avengers' } });
    const searchButton = screen.getByRole('button', { name: 'Search' });
    fireEvent.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText('Test Movie 1')).toBeInTheDocument();
    });
    const addButtons = screen.getAllByRole('button', { name: 'Add movie' });
    fireEvent.click(addButtons[0]);
    expect(screen.getByText('Selected Movies:')).toBeInTheDocument();
    expect(screen.getByText('Test Movie 1 (2023)')).toBeInTheDocument();
    fireEvent.click(addButtons[1]);
    expect(screen.getByText('Test Movie 2 (2023)')).toBeInTheDocument();
    const finishButton = screen.getByRole('button', { name: 'Finish Onboarding' });
    expect(finishButton).toBeDisabled();
    fireEvent.click(addButtons[0]);
    expect(finishButton).not.toBeDisabled();
  });

  test('completes onboarding process successfully', async () => {
    render(<OnboardingPage />);
    const searchInput = screen.getByPlaceholderText('Search for a movie...');
    fireEvent.change(searchInput, { target: { value: 'avengers' } });
    const searchButton = screen.getByRole('button', { name: 'Search' });
    fireEvent.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText('Test Movie 1')).toBeInTheDocument();
    });
    const addButtons = screen.getAllByRole('button', { name: 'Add movie' });
    fireEvent.click(addButtons[0]);
    fireEvent.click(addButtons[1]);
    fireEvent.click(addButtons[0]);
    const finishButton = screen.getByRole('button', { name: 'Finish Onboarding' });
    fireEvent.click(finishButton);
    await waitFor(() => {
      expect(mockStoreData.updateOnboarding).toHaveBeenCalledWith(['123', '456', '123']);
      expect(mockNavigate).toHaveBeenCalledWith('/dashboard');
    });
  });

  test('handles removing movies from favorites', async () => {
    render(<OnboardingPage />);
    const searchInput = screen.getByPlaceholderText('Search for a movie...');
    fireEvent.change(searchInput, { target: { value: 'avengers' } });
    const searchButton = screen.getByRole('button', { name: 'Search' });
    fireEvent.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText('Test Movie 1')).toBeInTheDocument();
    });
    const addButtons = screen.getAllByRole('button', { name: 'Add movie' });
    fireEvent.click(addButtons[0]);
    fireEvent.click(addButtons[1]);
    expect(screen.getByText('Test Movie 1 (2023)')).toBeInTheDocument();
    expect(screen.getByText('Test Movie 2 (2023)')).toBeInTheDocument();
    const removeButton = screen.getByRole('button', { name: 'Remove' });
    fireEvent.click(removeButton);
    expect(screen.queryByText('Test Movie 1 (2023)')).not.toBeInTheDocument();
    expect(screen.getByText('Test Movie 2 (2023)')).toBeInTheDocument();
  });

  test('handles logout functionality', () => {
    render(<OnboardingPage />);
    const logoutButton = screen.getByRole('button', { name: 'Logout' });
    fireEvent.click(logoutButton);
    expect(mockStoreData.logout).toHaveBeenCalled();
  });
});
